from .replace_operation import (
    replace_backward,
    ExtendOptimizerNames,
    __post_init__
)
from .bitsandbytes import (
    BitAdagrad,
    Adagrad8bit,
    Adagrad32bit,
    BitAdam,
    BitAdamW,
    Adam8bit,
    Adam32bit,
    PagedAdam, 
    PagedAdam8bit, 
    PagedAdam32bit,
    AdamW8bit,
    AdamW32bit,
    PagedAdamW,
    PagedAdamW8bit,
    PagedAdamW32bit,
    BitLAMB,
    LAMB8bit, 
    LAMB32bit,
    BitLARS,
    LARS8bit, 
    LARS32bit, 
    PytorchLARS,
    BitLion,
    Lion8bit, 
    Lion32bit, 
    PagedLion, 
    PagedLion8bit, 
    PagedLion32bit,
    BitRMSprop,
    RMSprop8bit, 
    RMSprop32bit,
    BitSGD,
    SGD8bit, 
    SGD32bit
)

from transformers import training_args

training_args.__post_init__ = __post_init__