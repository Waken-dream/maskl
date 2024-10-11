import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


class burgers_ON:
    '''
    Masked Training Network: recover origin buregrs equation data.
    '''
    def __init__(self, in_features, width):
        super(burgers_ON, self).__init__()