# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .adagrad import Adagrad as BitAdagrad
from .adagrad import Adagrad8bit, Adagrad32bit
from .adam import Adam as BitAdam
from .adam import Adam8bit, Adam32bit, PagedAdam, PagedAdam8bit, PagedAdam32bit
from .adamw import AdamW as BitAdamW 
from .adamw import (
    AdamW8bit,
    AdamW32bit,
    PagedAdamW,
    PagedAdamW8bit,
    PagedAdamW32bit,
)
from .lamb import LAMB as BitLAMB
from .lamb import LAMB8bit, LAMB32bit
from .lars import LARS as BitLARS
from .lars import LARS8bit, LARS32bit, PytorchLARS
from .lion import Lion as BitLion
from .lion import Lion8bit, Lion32bit, PagedLion, PagedLion8bit, PagedLion32bit
from ..optimizer import GlobalOptimManager
from .rmsprop import RMSprop as BitRMSprop
from .rmsprop import RMSprop8bit, RMSprop32bit
from .sgd import SGD as BitSGD
from .sgd import SGD8bit, SGD32bit
