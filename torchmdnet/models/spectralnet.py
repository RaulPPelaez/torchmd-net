import torch
from typing import Optional, Tuple
from torch import Tensor, nn
from torchmdnet.models.utils import (
    CosineCutoff,
    OptimizedDistance,
    rbf_class_mapping,
    act_class_mapping,
)
