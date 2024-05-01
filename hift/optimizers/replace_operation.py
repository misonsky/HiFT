import torch
import warnings
from collections import defaultdict
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
        if self.verbose:
            logger.info("[deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss "
                            "scale: {}, reducing to {}".format(prev_scale, self.cur_scale))
        return self.overflow

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
    
