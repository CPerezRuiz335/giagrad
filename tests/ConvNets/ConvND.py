import sys
sys.path.append('../../')
from giagrad import Tensor
import string
from itertools import chain
from math import floor
import numpy as np
import giagrad.nn as gnn
import giagrad.nn.layers.convs.utils as utils
# import torch

# torch.manual_seed(123)
np.random.seed(123)

##
print(
    utils.flat_tuple(
        ((1,2), 3, (1,4,2,5), (3,))
    )
)

print(
    utils.same_len(
            *((1,), (3,), (1,), (3,))
    )
)

print(
    utils.format_padding(5, 3)
)

utils.check_parameters(
    kernel_size=(3, 2, 2),
    stride=2,
    dilation=3,
    padding=(2,1, 9)
)

t = np.empty((12,12), dtype=np.bool_)
print(t)

print(
    utils.conv_output_shape(
        array_shape=(2, 3, 12, 12),
        kernel_size=(3, 3),
        stride=2,
        dilation=1,
        padding=((2,2), (2,2))
    )
)

print(
    utils.trimm_uneven_stride(
        array=t,
        kernel_size=(3, 3),
        stride=2,
        dilation=1
    )
)

print(
    utils.trans_output_shape(
        array_shape=(2, 3, 7, 7),
        kernel_size=(3, 3),
        stride=2,
        dilation=1,
        padding=((2,2), (2,2))
    )
)