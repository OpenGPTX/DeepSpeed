from types import MethodType
import torch
import deepspeed
import pytest
from deepspeed.ops.adam import FusedAdam
from tests.unit.common import DistributedTest
from op_builder import CPUAdamBuilder
from tests.unit.simple_model import SimpleModel, SimpleOptimizer, random_dataloader, LinearStackPipe, TiedLinearStackPipe
from tests.unit.util import bf16_required_version_check
from deepspeed import comm as dist
from deepspeed.runtime.pipe import schedule


class TestAdamBF16ZeroOneCycleCompatibility(DistributedTest):
    world_size = 1

    def test(self, zero_stage=2, use_cpu_offload=False):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "train_micro_batch_size_per_gpu": 1,
            "scheduler": {
                "type": "OneCycle",
                "params": {
                    "cycle_first_step_size": 16000,
                    "cycle_first_stair_count": 8000,
                    "decay_step_size": 16000,
                    "cycle_min_lr": 1e-06,
                    "cycle_max_lr": 3e-05,
                    "decay_lr_rate": 1e-07,
                    "cycle_min_mom": 0.85,
                    "cycle_max_mom": 0.99,
                    "decay_mom_rate": 0.0
                }
            },
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload
            }
        }

        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestZeroAllowUntestedOptimizer(DistributedTest):
    world_size = 1

    def test(self, zero_stage=2, use_cpu_offload=False):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_batch_size": 4,
            "steps_per_print": 1,
            "fp16": {
                "enabled": False,
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload
            },
            "zero_allow_untested_optimizer": False
        }

        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        optimizer = SimpleOptimizer(model.parameters())
        with pytest.raises(AssertionError):
            model, optim, _, _ = deepspeed.initialize(config=config_dict,
                                                      model=model,
                                                      optimizer=optimizer,
                                                      model_parameters=model.parameters())


class TestZeroEmptyPartition(DistributedTest):
    world_size = 3

    def test(self, zero_stage=2, use_cpu_offload=False):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        if zero_stage == 3:
            pytest.skip("skip for now")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": True
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload,
                "reduce_bucket_size": 100,
                "allgather_bucket_size": 100
            }
        }

        hidden_dim = 1
        model = SimpleModel(hidden_dim)

        # Ensure model has 2 parameters, to cause empty partition with DP=3
        assert len(list(model.parameters())) == 2
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              model_parameters=model.parameters())

        # Now make sure things work..
        data_loader = random_dataloader(model=model,
                                        total_samples=1,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.parametrize("optimizer_constructor", [torch.optim.Adam, FusedAdam])
class TestZeroSupportedClientOptimizer(DistributedTest):
    world_size = 1

    def test(self, optimizer_constructor, zero_stage=2):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        client_optimizer = optimizer_constructor(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              optimizer=client_optimizer)


class TestZero2ReduceScatterOff(DistributedTest):
    world_size = 2

    def test(self):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "allgather_bucket_size": 2000000000,
                "reduce_bucket_size": 200000000,
                "overlap_comm": False,
                "reduce_scatter": False
            },
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": True
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestZeroEmptyGrad(DistributedTest):
    world_size = 1

    def test(self, stage=2):
        if not bf16_required_version_check():
            pytest.skip(
                " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
            )

        config_dict = {
            "train_batch_size": 1,
            "steps_per_print": 1,
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": stage
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.Adam(model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              optimizer=optimizer)
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.bfloat16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.parametrize("comp_type",
                         [torch.float16,
                          torch.bfloat16,
                          torch.float],
                         ids=["fp16",
                              "bfp16",
                              "fp32"])
@pytest.mark.parametrize("comm_type",
                         [torch.float16,
                          torch.bfloat16],
                         ids=["fp16",
                              "bfp16"])
class TestZeroDtypeCocktail(DistributedTest):
    world_size = 2

    def test(self, comp_type, comm_type):
        if comp_type == torch.bfloat16 or comm_type == torch.bfloat16:
            if not bf16_required_version_check():
                pytest.skip(
                    " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
                )

        type_str = {torch.float16: "fp16", torch.bfloat16: "bfp16"}

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "fp16": {
                "enabled": comp_type == torch.float16
            },
            "bf16": {
                "enabled": comp_type == torch.bfloat16
            },
            "zero_optimization": {
                "stage": 2
            },
            "communication_data_type": type_str[comm_type]
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.Adam(model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict,
                                              model=model,
                                              optimizer=optimizer)
        data_loader = random_dataloader(model=model,
                                        total_samples=2,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=comp_type)

        def custom_reduce(tensor, dst, op=dist.ReduceOp.SUM, group=None, async_op=False):
            assert tensor.dtype == comm_type
            return orig_torch_reduce(tensor, dst, op, group, async_op)

        orig_torch_reduce = dist.reduce
        dist.reduce = custom_reduce
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
        dist.reduce = orig_torch_reduce


class TestBF16Training(DistributedTest):

    # TODO: Use @pytest.fixture
    def set_up(self, zero_stage: int):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage,
            },
            "communication_data_type": "fp32"
        }

        input_dim = 1
        hidden_dim = 10
        output_dim = 10
        num_layers = 4
        num_stages = 2

        pipe_model = LinearStackPipe(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_stages=num_stages,
        )
        optimizer = torch.optim.Adam(pipe_model.parameters())
        deepspeed_model, _, _, _ = deepspeed.initialize(
            config=config_dict,
            model=pipe_model,
            optimizer=optimizer,
        )

        self.model = deepspeed_model

    def set_up_tied_model(self, zero_stage: int):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "bf16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage
            },
            "communication_data_type": "fp32"
        }

        input_dim = 10
        hidden_dim = 10
        output_dim = 10
        num_layers = 4
        num_stages = 2

        pipe_model = TiedLinearStackPipe(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_stages=num_stages,
        )

        optimizer = torch.optim.Adam(pipe_model.parameters())
        deepspeed_model, _, _, _ = deepspeed.initialize(
            config=config_dict,
            model=pipe_model,
            optimizer=optimizer,
        )
        self.data_loader = random_dataloader(model=deepspeed_model,
                                             total_samples=2,
                                             hidden_dim=hidden_dim,
                                             device=deepspeed_model.device,
                                             dtype=torch.bfloat16)

        self.tied_model = deepspeed_model
        self.tied_model.set_dataloader(self.data_loader)

    def _check_params(self):
        params = list(self.model.parameters())

        for p in params:
            assert (p.dtype == torch.bfloat16)

    def test_parameter_type(self):
        self.set_up(zero_stage=0)
        self._check_params()
        self.set_up(zero_stage=1)
        self._check_params()

    def test_communication_data_type(self):
        self.set_up(zero_stage=0)
        assert (self.model.communication_data_type == torch.float32)

        self.set_up(zero_stage=1)
        assert (self.model.communication_data_type == torch.float32)

    def test__exec_reduce_tied_grads(self):
        # self.set_up_tied_model(1)
        self.set_up_tied_model(1)
        for n, batch in enumerate(self.data_loader):
            self.tied_model.module.train()
            self.tied_model.total_loss = None
            self.tied_model._compute_loss = True

            # Do the work
            self.tied_model.timers("train_batch").start()

            sched = schedule.TrainSchedule(
                micro_batches=self.tied_model.micro_batches,
                stages=self.tied_model.num_stages,
                stage_id=self.tied_model.stage_id,
            )
            # Reserve and reset buffers.
            self.tied_model._reserve_pipe_buffers(sched.num_pipe_buffers())
            self.tied_model.fwd_outputs = []
            for step_cmds in sched:
                # For each instruction in the step
                for cmd in step_cmds:
                    if type(cmd) not in self.tied_model._INSTRUCTION_MAP:
                        raise RuntimeError(
                            f"{self.__class__.__name__} does not understand instruction {repr(cmd)}"
                        )

                    # Equivalent to: self._exec_forward_pass(buffer_id=0)
                    self.tied_model._exec_instr = MethodType(self.tied_model._INSTRUCTION_MAP[type(cmd)], self.tied_model)
                    if type(cmd) == schedule.ReduceTiedGrads:
                        # check the gradient data types before and after executing ReduceTiedGrads
                        # during the execution it is not possible to access the gradients
                        weight_group_list = self.tied_model.module.get_tied_weights_and_groups()
                        for weight, group in weight_group_list:
                            assert weight.grad.dtype == torch.bfloat16
                        self.tied_model._exec_instr(**cmd.kwargs)
                        weight_group_list = self.tied_model.module.get_tied_weights_and_groups()
                        for weight, group in weight_group_list:
                            assert weight.grad.dtype == torch.bfloat16
                    else:
                        self.tied_model._exec_instr(**cmd.kwargs)
            break

    def test__exec_backward_pass(self):
        self.set_up_tied_model(0)
        for n, batch in enumerate(self.data_loader):
            self.tied_model.module.train()
            self.tied_model.total_loss = None
            self.tied_model._compute_loss = True

            # Do the work
            self.tied_model.timers("train_batch").start()

            sched = schedule.TrainSchedule(
                micro_batches=self.tied_model.micro_batches,
                stages=self.tied_model.num_stages,
                stage_id=self.tied_model.stage_id,
            )
            # Reserve and reset buffers.
            self.tied_model._reserve_pipe_buffers(sched.num_pipe_buffers())
            self.tied_model.fwd_outputs = []
            for step_cmds in sched:
                # For each instruction in the step
                for cmd in step_cmds:
                    if type(cmd) not in self.tied_model._INSTRUCTION_MAP:
                        raise RuntimeError(
                            f"{self.__class__.__name__} does not understand instruction {repr(cmd)}"
                        )

                    # Equivalent to: self._exec_forward_pass(buffer_id=0)
                    self.tied_model._exec_instr = MethodType(self.tied_model._INSTRUCTION_MAP[type(cmd)],
                                                             self.tied_model)
                    if type(cmd) == schedule.BackwardPass:
                        # check the gradient data types before and after executing ReduceTiedGrads
                        # during the execution it is not possible to access the gradients
                        self.tied_model._exec_instr(**cmd.kwargs)
                        if not self.tied_model.is_last_stage():
                            for group in self.tied_model.optimizer.bf16_groups:
                                for param in group:
                                    assert param.grad is None
                        print()
                    else:
                        self.tied_model._exec_instr(**cmd.kwargs)
            break
