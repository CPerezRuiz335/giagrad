import sys
sys.path.append('../../')
from giagrad import Tensor
import numpy as np
import giagrad.nn as gnn
import torch 
import torch.nn as tnn


BATCH = 2
IN_CHANNELS = 32
INPUT_SIZE = (5, 5)

OUT_CHANNELS = 4
KERNEL_SIZE = (3, 3)

STRIDE = (1, 1)
DILATION = (1, 1)
PADDING = (0, 0)
GROUPS = 4

conv_gg = gnn.Conv2D(
        out_channels=OUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        dilation=DILATION,
        padding=PADDING,
        groups=GROUPS,
        bias=True
    ) 

X = Tensor.empty(
	BATCH, IN_CHANNELS, *INPUT_SIZE, 
	requires_grad=True
).uniform(-4, 4)


out = conv_gg(X)
print(conv_gg.w.shape)

print('b shape', conv_gg.b.shape)

print(out.shape)

X_torch = torch.from_numpy(X.data).requires_grad_()

conv_torch = tnn.Conv2d(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        dilation=DILATION,
        padding=PADDING,
        groups=GROUPS,
        bias=True
    )


out_torch = conv_torch(X_torch)
print(out_torch.shape)