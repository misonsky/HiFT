import os
import io
import json
import torch
import warnings
from collections import defaultdict
from pathlib import Path
from transformers.utils.generic import strtobool
import deepspeed
from deepspeed.moe.utils import split_params_grads_into_shared_and_expert_params
from functools import partial
from torch._utils import _flatten_dense_tensors

from deepspeed.runtime import DeepSpeedOptimizer
from deepspeed.runtime.utils import get_global_norm, CheckOverflow, get_weight_norm, required_torch_version
from deepspeed.runtime.fp16.loss_scaler import INITIAL_LOSS_SCALE, SCALE_WINDOW, MIN_LOSS_SCALE
from deepspeed.utils import logger
from deepspeed.checkpoint.constants import OPTIMIZER_STATE_DICT
from deepspeed.accelerator import get_accelerator
from deepspeed import comm as dist

from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer

from transformers.debug_utils import DebugOption
from huggingface_hub import get_full_repo_name
from packaging import version
from transformers import training_args
from transformers.training_args import (
    default_logdir,
    get_xla_device_type
)
from transformers.utils import (
    ExplicitEnum,
    cached_property,
    is_accelerate_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_bf16_cpu_available,
    is_torch_bf16_gpu_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_tf32_available,
    is_torch_tpu_available,
    is_torch_xpu_available,
    logging,
    requires_backends,
)
from transformers.trainer_utils import (
    EvaluationStrategy,
    FSDPOption,
    HubStrategy,
    IntervalStrategy,
    SchedulerType,
)
class ExtendOptimizerNames(ExplicitEnum):
    ADAMW_HF = "adamw_hf"
    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_FUSED = "adamw_torch_fused"
    ADAMW_TORCH_XLA = "adamw_torch_xla"
    ADAMW_TORCH_NPU_FUSED = "adamw_torch_npu_fused"
    ADAMW_APEX_FUSED = "adamw_apex_fused"
    ADAFACTOR = "adafactor"
    ADAMW_ANYPRECISION = "adamw_anyprecision"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    ADAMW_BNB = "adamw_bnb_8bit"
    ADAMW_8BIT = "adamw_8bit"  # just an alias for adamw_bnb_8bit
    LION_8BIT = "lion_8bit"
    LION = "lion_32bit"
    PAGED_ADAMW = "paged_adamw_32bit"
    PAGED_ADAMW_8BIT = "paged_adamw_8bit"
    PAGED_LION = "paged_lion_32bit"
    PAGED_LION_8BIT = "paged_lion_8bit"
    RMSPROP = "rmsprop"
    #extend
    ADAM = "adam"
    ADAM_32BIT = "adam_32bit"
    ADAM_8BIT = "adam_8bit"
    PAGED_ADAM = "paged_adam"
    PAGED_ADAM_32BIT = "paged_adam_32bit"
    PAGED_ADAM_8BIT = "paged_adam_8bit"
    LAMB ="lamb"
    LAMB_32BIT = "lamb_32bit"
    LAMB_8BIT = "lamb_8bit"
    LARS = "lars"
    LARS_32BIT = "lars_32bit"
    LARS_8BIT = "lars_8bit"
    BRMSPROP = "rmsprop_bit"
    BRMSPROP_32BIT = "rmsprop_32bit"
    BRMSPROP_8BIT = "rmsprop_8bit"
    BSGD = "sgd_bit"
    BSGD_32BIT = "sgd_32bit"
    BSGD_8BIT = "sgd_8bit"



def __init__(self,
    init_optimizer,
    deepspeed=None,
    static_loss_scale=1.0,
    dynamic_loss_scale=False,
    dynamic_loss_args=None,
    verbose=True,
    mpu=None,
    clip_grad=0.0,
    fused_lamb_legacy=False):
    self.fused_lamb_legacy = fused_lamb_legacy
    self._global_grad_norm = 0.

    if dist.get_rank() == 0:
        logger.info(f'Fused Lamb Legacy : {self.fused_lamb_legacy} ')
    
    if not get_accelerator().is_available():
        raise SystemError("Cannot use fp16 without accelerator.")
    self.optimizer = init_optimizer
    # param groups
    self.fp16_groups = defaultdict(dict)
    self.fp32_groups = defaultdict(dict)
    # loop to deal with groups
    for i, param_group in enumerate(self.optimizer.param_groups):
        #fp16 weights that represents the actual model weights
        self.fp16_groups[i]  = {id(p):p for p in param_group['params']}
        # self.fp16_groups.append(param_group['params'])
        #creating a fp32 copy of the weights that will be updated first then
        #copied to fp16 weights
        fp32_group = {id(p):p.clone().float().detach().to("cpu") for p in param_group['params']}
        #in case the internal optimizer needs it
        for p_id in fp32_group:
            fp32_group[p_id].requires_grad = False
        # fp32_group[id(param_group['params'][-1])].requires_grad = True
        self.fp32_groups[i] = fp32_group
        # param_group['params'] = [self.fp32_groups[i][id(param_group['params'][-1])].to(self.fp16_groups[i][id(param_group['params'][-1])].device)]
        
    if dynamic_loss_scale:
        self.dynamic_loss_scale = True
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = 2.0
        if dynamic_loss_args is None:
            self.cur_scale = 1.0 * 2**16
            self.scale_window = 1000
            self.min_loss_scale = 0.25
        else:
            self.cur_scale = dynamic_loss_args[INITIAL_LOSS_SCALE]
            self.scale_window = dynamic_loss_args[SCALE_WINDOW]
            self.min_loss_scale = dynamic_loss_args[MIN_LOSS_SCALE]
    else:
        self.dynamic_loss_scale = False
        self.cur_iter = 0
        self.cur_scale = static_loss_scale
    
    self.custom_loss_scaler = False
    self.external_loss_scale = None

    self.verbose = verbose

    self.clip_grad = clip_grad
    self.norm_type = 2

    if required_torch_version(max_version=0.4):
        self.clip_grad_norm = torch.nn.utils.clip_grad_norm
    else:
        self.clip_grad_norm = torch.nn.utils.clip_grad_norm_
    
    self.mpu = mpu

    self.overflow = False

    self.overflow_checker = CheckOverflow(mpu=self.mpu, deepspeed=deepspeed)

    # self.initialize_optimizer_states()
def zero_grad(self, set_to_none=True):
    """
    Zero FP16 parameter grads.
    """
    # FP32 grad should never exist outside of the step function
    # For speed, set model fp16 grad to None by default
    for group_num in self.fp16_groups:
        for p_id in self.fp16_groups[group_num]:
            p = self.fp16_groups[group_num][p_id]
            if set_to_none:
                p.grad = None
            else:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

def move_device(self,fp16_groups,fp32_groups):
    for group_num in self.fp32_groups:
        for p_id in self.fp32_groups[group_num]:
            p = self.fp32_groups[group_num][p_id]
            p.requires_grad = False
            if p.is_cuda:
                self.fp32_groups[group_num][p_id] = p.to("cpu")
                del p
    for i, param_group in enumerate(self.optimizer.param_groups):
        fp16_groups.append(param_group['params'])
        fp32_groups.append([self.fp32_groups[i][id(p)].to(p.device) for p in param_group['params']])
        for p in fp32_groups[i]:
            p.requires_grad = True
        for p_index,p in enumerate(param_group['params']):
            self.fp32_groups[i][id(p)] = fp32_groups[i][p_index]
            self.optimizer.add_id_mapping({fp32_groups[i][p_index]:id(p)})
        param_group['params'] = fp32_groups[i]
    torch.cuda.empty_cache()
    return fp16_groups,fp32_groups
def step(self, closure=None):
    """
    Not supporting closure.
    """
    fp16_groups = []
    fp32_groups = []
    self.move_device(fp16_groups,fp32_groups)
    if self.fused_lamb_legacy:
        return self.step_fused_lamb()
    self.overflow = self.overflow_checker.check(param_groups = fp16_groups)
    prev_scale = self.cur_scale

    self._update_scale(self.overflow)
    if self.overflow:
        for fp16_group in fp16_groups:
            for p in fp16_group:
                noise = torch.randn_like(p.grad,dtype=p.grad.dtype)
                p.grad = noise.to(p.device)
        if self.verbose:
            logger.info("[deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss "
                            "scale: {}, reducing to {}".format(prev_scale, self.cur_scale))
        # return self.overflow

    norm_groups = []
    for i, group in enumerate(fp16_groups):
        grads_for_norm, _ = split_params_grads_into_shared_and_expert_params(group)
        norm_group_value = 0.0
        if len(grads_for_norm) > 0:
            norm_group_value = get_weight_norm(grads_for_norm, mpu=self.mpu)
        norm_groups.append(norm_group_value)

        # copying gradients to fp32 to work with fp32 parameters
        for fp32_param, fp16_param in zip(fp32_groups[i], fp16_groups[i]):
            if fp16_param.grad is not None:
                fp32_param.grad = fp16_param.grad.to(fp32_param.dtype)

    self._global_grad_norm = get_global_norm(norm_list=norm_groups)
    self.unscale_and_clip_grads(total_norm=self._global_grad_norm,fp32_groups=fp32_groups)
    
    self.optimizer.step()
    self.optimizer.clear_id_mapping()
    for fp32_group, fp16_group in zip(fp32_groups, fp16_groups):
        for idx, (fp32_param, fp16_param) in enumerate(zip(fp32_group, fp16_group)):

            #remove the fp32 grad
            fp32_param.grad = None
            # fp32_param.requires_grad = False

            #copy data from fp32 to fp16
            fp16_param.data.copy_(fp32_param.data)
    return self.overflow

def unscale_and_clip_grads(self, total_norm, fp32_groups,apply_scale=True):
        # compute combined scale factor for this group
        combined_scale = self.cur_scale
        if self.clip_grad > 0.:
            # norm is in fact norm*scale
            clip = ((total_norm / self.cur_scale) + 1e-6) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.cur_scale

        if apply_scale:
            for group in fp32_groups:
                for param in group:
                    if param.grad is not None:
                        param.grad.data.mul_(1. / combined_scale)

        return combined_scale
def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['cur_scale'] = self.cur_scale
        state_dict['cur_iter'] = self.cur_iter
        if state_dict['dynamic_loss_scale']:
            state_dict['last_overflow_iter'] = self.last_overflow_iter
            state_dict['scale_factor'] = self.scale_factor
            state_dict['scale_window'] = self.scale_window
        state_dict[OPTIMIZER_STATE_DICT] = self.optimizer.state_dict()
        state_dict['fp32_groups'] = [list(d_e.values()) for d_e in [self.fp32_groups[i] for i in self.fp32_groups]]
        return state_dict

def replace_backward():
    logger.info("...[deepspeed] mixed precision adapted for HiFT are running......")
    FP16_UnfusedOptimizer.__init__ = __init__
    FP16_UnfusedOptimizer.zero_grad = zero_grad
    FP16_UnfusedOptimizer.move_device = move_device
    FP16_UnfusedOptimizer.step = step
    FP16_UnfusedOptimizer.unscale_and_clip_grads =unscale_and_clip_grads
    FP16_UnfusedOptimizer.state_dict = state_dict



def __post_init__(self):
        # expand paths, if not os.makedirs("~/bar") will make directory
        # in the current directory instead of the actual home
        # see https://github.com/huggingface/transformers/issues/10628
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, default_logdir())
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)

        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN

        if isinstance(self.evaluation_strategy, EvaluationStrategy):
            warnings.warn(
                "using `EvaluationStrategy` for `evaluation_strategy` is deprecated and will be removed in version 5"
                " of 🤗 Transformers. Use `IntervalStrategy` instead",
                FutureWarning,
            )
            # Go back to the underlying string or we won't be able to instantiate `IntervalStrategy` on it.
            self.evaluation_strategy = self.evaluation_strategy.value
        if self.no_cuda:
            warnings.warn(
                "using `no_cuda` is deprecated and will be removed in version 5.0 of 🤗 Transformers. "
                "Use `use_cpu` instead",
                FutureWarning,
            )
            self.use_cpu = self.no_cuda

        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)
        self.hub_strategy = HubStrategy(self.hub_strategy)

        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        if self.do_eval is False and self.evaluation_strategy != IntervalStrategy.NO:
            self.do_eval = True

        # eval_steps has to be defined and non-zero, fallbacks to logging_steps if the latter is non-zero
        if self.evaluation_strategy == IntervalStrategy.STEPS and (self.eval_steps is None or self.eval_steps == 0):
            if self.logging_steps > 0:
                logger.info(f"using `logging_steps` to initialize `eval_steps` to {self.logging_steps}")
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.evaluation_strategy} requires either non-zero --eval_steps or"
                    " --logging_steps"
                )

        # logging_steps must be non-zero for logging_strategy that is other than 'no'
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f"logging strategy {self.logging_strategy} requires non-zero --logging_steps")

        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps > 1:
            if self.logging_steps != int(self.logging_steps):
                raise ValueError(f"--logging_steps must be an integer if bigger than 1: {self.logging_steps}")
            self.logging_steps = int(self.logging_steps)
        if self.evaluation_strategy == IntervalStrategy.STEPS and self.eval_steps > 1:
            if self.eval_steps != int(self.eval_steps):
                raise ValueError(f"--eval_steps must be an integer if bigger than 1: {self.eval_steps}")
            self.eval_steps = int(self.eval_steps)
        if self.save_strategy == IntervalStrategy.STEPS and self.save_steps > 1:
            if self.save_steps != int(self.save_steps):
                raise ValueError(f"--save_steps must be an integer if bigger than 1: {self.save_steps}")
            self.save_steps = int(self.save_steps)

        # Sanity checks for load_best_model_at_end: we require save and eval strategies to be compatible.
        if self.load_best_model_at_end:
            if self.evaluation_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
                    f"strategy: {self.evaluation_strategy}\n- Save strategy: {self.save_strategy}"
                )
            if self.evaluation_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
                if self.eval_steps < 1 or self.save_steps < 1:
                    if not (self.eval_steps < 1 and self.save_steps < 1):
                        raise ValueError(
                            "--load_best_model_at_end requires the saving steps to be a multiple of the evaluation "
                            "steps, which cannot get guaranteed when mixing ratio and absolute steps for save_steps "
                            f"{self.save_steps} and eval_steps {self.eval_steps}."
                        )
                    # Work around floating point precision issues
                    LARGE_MULTIPLIER = 1_000_000
                    if (self.save_steps * LARGE_MULTIPLIER) % (self.eval_steps * LARGE_MULTIPLIER) != 0:
                        raise ValueError(
                            "--load_best_model_at_end requires the saving steps to be a multiple of the evaluation "
                            f"steps, but found {self.save_steps}, which is not a multiple of {self.eval_steps}."
                        )
                raise ValueError(
                    "--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation "
                    f"steps, but found {self.save_steps}, which is not a round multiple of {self.eval_steps}."
                )

        safetensors_available = is_safetensors_available()
        if self.save_safetensors and not safetensors_available:
            raise ValueError(f"--save_safetensors={self.save_safetensors} requires safetensors to be installed!")
        if not self.save_safetensors and safetensors_available:
            logger.info(
                f"Found safetensors installation, but --save_safetensors={self.save_safetensors}. "
                f"Safetensors should be a preferred weights saving format due to security and performance reasons. "
                f"If your model cannot be saved by safetensors please feel free to open an issue at "
                f"https://github.com/huggingface/safetensors!"
            )

        if (
            self.load_best_model_at_end or self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU
        ) and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in ["loss", "eval_loss"]
        if self.run_name is None:
            self.run_name = self.output_dir
        if self.framework == "pt" and is_torch_available():
            if self.fp16_backend and self.fp16_backend != "auto":
                warnings.warn(
                    "`fp16_backend` is deprecated and will be removed in version 5 of 🤗 Transformers. Use"
                    " `half_precision_backend` instead",
                    FutureWarning,
                )
                self.half_precision_backend = self.fp16_backend

            if self.bf16 or self.bf16_full_eval:
                if self.use_cpu and not is_torch_bf16_cpu_available() and not is_torch_tpu_available():
                    # cpu
                    raise ValueError("Your setup doesn't support bf16/(cpu, tpu, neuroncore). You need torch>=1.10")
                elif not self.use_cpu:
                    if torch.cuda.is_available() and not is_torch_bf16_gpu_available():
                        # gpu
                        raise ValueError(
                            "Your setup doesn't support bf16/gpu. You need torch>=1.10, using Ampere GPU with cuda>=11.0"
                        )
                    elif is_torch_npu_available():
                        # npu
                        from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_11

                        if not is_torch_greater_or_equal_than_1_11:
                            raise ValueError(
                                "Your setup doesn't support bf16/npu. You need torch>=1.11, using Ascend NPU with "
                                "`torch_npu` installed"
                            )
                    elif not is_torch_xpu_available():
                        # xpu
                        from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_12

                        if not is_torch_greater_or_equal_than_1_12:
                            raise ValueError(
                                "Your setup doesn't support bf16/xpu. You need torch>=1.12, using Intel XPU/GPU with IPEX installed"
                            )

        if self.fp16 and self.bf16:
            raise ValueError("At most one of fp16 and bf16 can be True, but not both")

        if self.fp16_full_eval and self.bf16_full_eval:
            raise ValueError("At most one of fp16 and bf16 can be True for full eval, but not both")

        if self.bf16:
            if self.half_precision_backend == "apex":
                raise ValueError(" `--half_precision_backend apex`: GPU bf16 is not supported by apex.")

        if self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            if self.evaluation_strategy == IntervalStrategy.NO:
                raise ValueError("lr_scheduler_type reduce_lr_on_plateau requires an eval strategy")
            if not is_torch_available():
                raise ValueError("lr_scheduler_type reduce_lr_on_plateau requires torch>=0.2.0")

        self.optim = ExtendOptimizerNames(self.optim)
        if self.adafactor:
            warnings.warn(
                "`--adafactor` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--optim"
                " adafactor` instead",
                FutureWarning,
            )
            self.optim = ExtendOptimizerNames.ADAFACTOR
        if self.optim == ExtendOptimizerNames.ADAMW_TORCH_FUSED and is_torch_available():
            if version.parse(version.parse(torch.__version__).base_version) < version.parse("2.0.0"):
                raise ValueError("--optim adamw_torch_fused requires PyTorch 2.0 or higher")
            # there is a bug in fp16/AMP in pt-2.0.0
            if version.parse(version.parse(torch.__version__).base_version) == version.parse("2.0.0") and self.fp16:
                raise ValueError("--optim adamw_torch_fused with --fp16 requires PyTorch>2.0")

        if (
            self.framework == "pt"
            and is_torch_available()
            and (self.device.type != "cuda")
            and (self.device.type != "npu")
            and (self.device.type != "xpu")
            and (get_xla_device_type(self.device) != "GPU")
            and (self.fp16 or self.fp16_full_eval)
        ):
            raise ValueError(
                "FP16 Mixed precision training with AMP or APEX (`--fp16`) and FP16 half precision evaluation"
                " (`--fp16_full_eval`) can only be used on CUDA or NPU devices or certain XPU devices (with IPEX)."
            )

        if (
            self.framework == "pt"
            and is_torch_available()
            and (self.device.type != "cuda")
            and (self.device.type != "npu")
            and (self.device.type != "xpu")
            and (get_xla_device_type(self.device) != "GPU")
            and (get_xla_device_type(self.device) != "TPU")
            and (self.device.type != "cpu")
            and (self.bf16 or self.bf16_full_eval)
        ):
            raise ValueError(
                "BF16 Mixed precision training with AMP (`--bf16`) and BF16 half precision evaluation"
                " (`--bf16_full_eval`) can only be used on CUDA, XPU (with IPEX), NPU or CPU/TPU/NeuronCore devices."
            )

        if self.torchdynamo is not None:
            warnings.warn(
                "`torchdynamo` is deprecated and will be removed in version 5 of 🤗 Transformers. Use"
                " `torch_compile_backend` instead",
                FutureWarning,
            )
            self.torch_compile_backend = self.torchdynamo
        if (self.torch_compile_mode is not None or self.torch_compile_backend is not None) and not self.torch_compile:
            self.torch_compile = True
        if self.torch_compile and self.torch_compile_backend is None:
            self.torch_compile_backend = "inductor"

        # accelerate integration for torch compile
        if self.torch_compile:
            # set env vars for accelerate
            prefix = "ACCELERATE_DYNAMO_"
            os.environ[prefix + "BACKEND"] = self.torch_compile_backend
            if self.torch_compile_mode is not None:
                os.environ[prefix + "MODE"] = self.torch_compile_mode

        if self.framework == "pt" and is_torch_available() and self.torch_compile:
            if is_torch_tf32_available():
                if self.tf32 is None and not self.fp16 or self.bf16:
                    logger.info(
                        "Setting TF32 in CUDA backends to speedup torch compile, you won't see any improvement"
                        " otherwise."
                    )
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            else:
                logger.warning(
                    "The speedups for torchdynamo mostly come wih GPU Ampere or higher and which is not detected here."
                )
        if self.framework == "pt" and is_torch_available() and self.tf32 is not None:
            if self.tf32:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                else:
                    raise ValueError("--tf32 requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7")
            else:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                # no need to assert on else

        # if training args is specified, it will override the one specified in the accelerate config
        if self.half_precision_backend != "apex":
            mixed_precision_dtype = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
            if self.fp16:
                mixed_precision_dtype = "fp16"
            elif self.bf16:
                mixed_precision_dtype = "bf16"
            os.environ["ACCELERATE_MIXED_PRECISION"] = mixed_precision_dtype

        if self.report_to is None:
            logger.info(
                "The default value for the training argument `--report_to` will change in v5 (from all installed "
                "integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as "
                "now. You should start updating your code and make this info disappear :-)."
            )
            self.report_to = "all"
        if self.report_to == "all" or self.report_to == ["all"]:
            # Import at runtime to avoid a circular import.
            from transformers.integrations import get_available_reporting_integrations

            self.report_to = get_available_reporting_integrations()
        elif self.report_to == "none" or self.report_to == ["none"]:
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logger.info(
                "Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio"
                " during training"
            )

        if isinstance(self.fsdp, bool):
            self.fsdp = "full_shard" if self.fsdp else ""
        if isinstance(self.fsdp, str):
            self.fsdp = [FSDPOption(s) for s in self.fsdp.split()]
        if self.fsdp == [FSDPOption.OFFLOAD]:
            raise ValueError(
                "`--fsdp offload` can't work on its own. It needs to be added to `--fsdp full_shard` or "
                '`--fsdp shard_grad_op`. For example, `--fsdp "full_shard offload"`.'
            )
        elif FSDPOption.FULL_SHARD in self.fsdp and FSDPOption.SHARD_GRAD_OP in self.fsdp:
            raise ValueError("`--fsdp full_shard` is not compatible with `--fsdp shard_grad_op`.")

        if self.fsdp_config is None:
            self.fsdp_config = {}

        if isinstance(self.fsdp_config, str):
            if len(self.fsdp) == 0:
                warnings.warn("`--fsdp_config` is useful only when `--fsdp` is specified.")
            with io.open(self.fsdp_config, "r", encoding="utf-8") as f:
                self.fsdp_config = json.load(f)
                for k in list(self.fsdp_config.keys()):
                    if k.startswith("fsdp_"):
                        v = self.fsdp_config.pop(k)
                        self.fsdp_config[k[5:]] = v

        if self.fsdp_min_num_params > 0:
            warnings.warn("using `--fsdp_min_num_params` is deprecated. Use fsdp_config instead ", FutureWarning)

        self.fsdp_config["min_num_params"] = max(self.fsdp_config.get("min_num_params", 0), self.fsdp_min_num_params)

        # if fsdp_config["transformer_layer_cls_to_wrap"] is specified as a string, convert it to a list with a single object
        if isinstance(self.fsdp_config.get("transformer_layer_cls_to_wrap", None), str):
            self.fsdp_config["transformer_layer_cls_to_wrap"] = [self.fsdp_config["transformer_layer_cls_to_wrap"]]

        if self.fsdp_transformer_layer_cls_to_wrap is not None:
            warnings.warn(
                "using `--fsdp_transformer_layer_cls_to_wrap` is deprecated. Use fsdp_config instead ", FutureWarning
            )
            self.fsdp_config["transformer_layer_cls_to_wrap"] = self.fsdp_config.get(
                "transformer_layer_cls_to_wrap", []
            ) + [self.fsdp_transformer_layer_cls_to_wrap]

        if len(self.fsdp) == 0 and self.fsdp_config["min_num_params"] > 0:
            warnings.warn("`min_num_params` is useful only when `--fsdp` is specified.")

        if len(self.fsdp) == 0 and self.fsdp_config.get("transformer_layer_cls_to_wrap", None) is not None:
            warnings.warn("`transformer_layer_cls_to_wrap` is useful only when `--fsdp` is specified.")

        if (
            len(self.fsdp) > 0
            and self.fsdp_config["min_num_params"] > 0
            and self.fsdp_config.get("transformer_layer_cls_to_wrap", None) is not None
        ):
            raise ValueError("`min_num_params` and `transformer_layer_cls_to_wrap` are mutually exclusive.")
        self.fsdp_config["xla"] = self.fsdp_config.get("xla", False)
        self.fsdp_config["xla_fsdp_grad_ckpt"] = self.fsdp_config.get("xla_fsdp_grad_ckpt", False)
        if self.fsdp_config["xla"]:
            if len(self.fsdp) > 0:
                # store XLA fsdp configuration parameters into a dictionary
                self.xla_fsdp_config = self.fsdp_config.get("xla_fsdp_settings", {})
                # apply appropriate string to torch.dtype conversions for parameters
                if "compute_dtype" in self.xla_fsdp_config:
                    self.xla_fsdp_config["compute_dtype"] = getattr(torch, self.xla_fsdp_config["compute_dtype"])
                if "buffer_dtype" in self.xla_fsdp_config:
                    self.xla_fsdp_config["buffer_dtype"] = getattr(torch, self.xla_fsdp_config["buffer_dtype"])
            else:
                warnings.warn("XLA FSDP can be used only when `--fsdp` is specified.")
        else:
            if self.fsdp_config["xla_fsdp_grad_ckpt"]:
                warnings.warn("`--xla_fsdp_grad_ckpt` is useful only when `--xla` is set to true.")

        # accelerate integration for FSDP
        if len(self.fsdp) > 0 and not self.fsdp_config["xla"]:
            os.environ["ACCELERATE_USE_FSDP"] = "true"
            from accelerate.utils.constants import (
                FSDP_AUTO_WRAP_POLICY,
                FSDP_SHARDING_STRATEGY,
            )

            prefix = "FSDP_"
            for fsdp_option in self.fsdp:
                if fsdp_option.upper() in FSDP_SHARDING_STRATEGY:
                    # set environment variable for FSDP sharding strategy
                    os.environ[f"{prefix}SHARDING_STRATEGY"] = str(
                        FSDP_SHARDING_STRATEGY.index(fsdp_option.upper()) + 1
                    )
                elif fsdp_option == FSDPOption.OFFLOAD:
                    os.environ[f"{prefix}OFFLOAD_PARAMS"] = "true"
                elif fsdp_option == FSDPOption.AUTO_WRAP:
                    os.environ[f"{prefix}AUTO_WRAP_POLICY"] = FSDP_AUTO_WRAP_POLICY[0]
                    if self.fsdp_config["min_num_params"] > 0:
                        os.environ[f"{prefix}MIN_NUM_PARAMS"] = str(self.fsdp_config["min_num_params"])
                        os.environ[f"{prefix}AUTO_WRAP_POLICY"] = FSDP_AUTO_WRAP_POLICY[1]
                    elif self.fsdp_config.get("transformer_layer_cls_to_wrap", None) is not None:
                        os.environ[f"{prefix}TRANSFORMER_CLS_TO_WRAP"] = ",".join(
                            self.fsdp_config["transformer_layer_cls_to_wrap"]
                        )
            prefetch_policy = self.fsdp_config.get("fsdp_backward_prefetch", "NO_PREFETCH")
            os.environ[f"{prefix}BACKWARD_PREFETCH"] = prefetch_policy.upper()
            os.environ[f"{prefix}FORWARD_PREFETCH"] = self.fsdp_config.get("forward_prefect", "false")
            os.environ[f"{prefix}SYNC_MODULE_STATES"] = self.fsdp_config.get("sync_module_states", "true")
            os.environ[f"{prefix}USE_ORIG_PARAMS"] = self.fsdp_config.get("use_orig_params", "true")

        if self.tpu_metrics_debug:
            warnings.warn(
                "using `--tpu_metrics_debug` is deprecated and will be removed in version 5 of 🤗 Transformers. Use"
                " `--debug tpu_metrics_debug` instead",
                FutureWarning,
            )
            if self.debug is None:
                self.debug = " tpu_metrics_debug"
            else:
                self.debug += " tpu_metrics_debug"
            self.tpu_metrics_debug = False

        if isinstance(self.debug, str):
            self.debug = [DebugOption(s) for s in self.debug.split()]
        elif self.debug is None:
            self.debug = []

        self.deepspeed_plugin = None
        if self.deepspeed:
            # - must be run very last in arg parsing, since it will use a lot of these settings.
            # - must be run before the model is created.
            if not is_accelerate_available():
                raise ValueError("--deepspeed requires Accelerate to be installed: `pip install accelerate`.")
            from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

            # will be used later by the Trainer
            # note: leave self.deepspeed unmodified in case a user relies on it not to be modified)
            self.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.deepspeed)
            self.hf_deepspeed_config.trainer_config_process(self)

            # Accelerate DeepSpeed Plugin
            from accelerate.utils import DeepSpeedPlugin

            os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
            self.deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.hf_deepspeed_config)
        elif strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")):
            # Accelerate DeepSpeed Plugin
            from accelerate.utils import DeepSpeedPlugin

            self.deepspeed_plugin = DeepSpeedPlugin()
            mixed_precision = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
            self.deepspeed_plugin.set_mixed_precision(mixed_precision)
            self.deepspeed_plugin.set_deepspeed_weakref()

        if self.use_cpu:
            self.dataloader_pin_memory = False

        if self.push_to_hub_token is not None:
            warnings.warn(
                "`--push_to_hub_token` is deprecated and will be removed in version 5 of 🤗 Transformers. Use "
                "`--hub_token` instead.",
                FutureWarning,
            )
            self.hub_token = self.push_to_hub_token

        if self.push_to_hub_model_id is not None:
            self.hub_model_id = get_full_repo_name(
                self.push_to_hub_model_id, organization=self.push_to_hub_organization, token=self.hub_token
            )
            if self.push_to_hub_organization is not None:
                warnings.warn(
                    "`--push_to_hub_model_id` and `--push_to_hub_organization` are deprecated and will be removed in "
                    "version 5 of 🤗 Transformers. Use `--hub_model_id` instead and pass the full repo name to this "
                    f"argument (in this case {self.hub_model_id}).",
                    FutureWarning,
                )
            else:
                warnings.warn(
                    "`--push_to_hub_model_id` is deprecated and will be removed in version 5 of 🤗 Transformers. Use "
                    "`--hub_model_id` instead and pass the full repo name to this argument (in this case "
                    f"{self.hub_model_id}).",
                    FutureWarning,
                )
        elif self.push_to_hub_organization is not None:
            self.hub_model_id = f"{self.push_to_hub_organization}/{Path(self.output_dir).name}"
            warnings.warn(
                "`--push_to_hub_organization` is deprecated and will be removed in version 5 of 🤗 Transformers. Use "
                "`--hub_model_id` instead and pass the full repo name to this argument (in this case "
                f"{self.hub_model_id}).",
                FutureWarning,
            )

