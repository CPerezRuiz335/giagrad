import sys
sys.path.append('../')
from giagrad import Tensor
import numpy as np
import giagrad.nn as gnn
import torch
import torch.nn as tnn

check = input(
    'comment __init_tensors in giagrad.nn.layers.convnd, done?: [T, F] -> '
)
if check != 'T': 
    raise ValueError('test not valid')

BATCH = 2
KERNEL_SIZE = (3, 2, 2)
INPUT_SIZE = (5, 5, 5)
IN_CHANNELS = 3 
OUT_CHANNELS = 4 
STRIDE = (2, 3, 1)
DILATION = (2, 1, 2)
PADDING = (3, 2, 1)


X_torch = torch.rand(
        BATCH, IN_CHANNELS, *INPUT_SIZE,
        requires_grad=True
    ) 

X_gg = Tensor(
    X_torch.detach().numpy(),
    requires_grad=True
    )

conv_torch = tnn.Conv3d(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        dilation=DILATION,
        padding=PADDING,
        bias=False
    )

conv_gg = gnn.Conv3D(
        out_channels=OUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        dilation=DILATION,
        padding=PADDING,
        bias=False
    ) 

conv_gg.w = Tensor(
    conv_torch.weight.detach().numpy(),
    requires_grad=True,
)

out_torch = conv_torch(X_torch)
out_gg = conv_gg(X_gg)

out_torch.sum().backward()
out_gg.sum().backward()

tol = 1e-4
assert np.all(
    abs(out_gg.data 
        - out_torch.detach().numpy()
    ) < tol
)

assert np.all(
    abs(conv_gg.w.grad 
        - conv_torch.weight.grad.detach().numpy()
    ) < tol
)

assert np.all(
    abs(X_gg.grad 
        - X_torch.grad.detach().numpy()
    ) < tol
)

BATCH = 5
KERNEL_SIZE = (3, 3)
INPUT_SIZE = (10, 10)
IN_CHANNELS = 3 
OUT_CHANNELS = 4 
STRIDE = 3
DILATION = 2
PADDING = 5


X_torch = torch.rand(
        BATCH, IN_CHANNELS, *INPUT_SIZE,
        requires_grad=True
    ) 

X_gg = Tensor(
    X_torch.detach().numpy(),
    requires_grad=True
    )

conv_torch = tnn.Conv2d(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        dilation=DILATION,
        padding=PADDING,
        bias=False
    )

conv_gg = gnn.Conv2D(
        out_channels=OUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        dilation=DILATION,
        padding=PADDING,
        bias=False
    ) 

conv_gg.w = Tensor(
    conv_torch.weight.detach().numpy(),
    requires_grad=True,
)

out_torch = conv_torch(X_torch)
out_gg = conv_gg(X_gg)

out_torch.sum().backward()
out_gg.sum().backward()

tol = 1e-5
assert np.all(
    abs(out_gg.data 
        - out_torch.detach().numpy()
    ) < tol
)

assert np.all(
    abs(conv_gg.w.grad 
        - conv_torch.weight.grad.detach().numpy()
    ) < tol
)

assert np.all(
    abs(X_gg.grad 
        - X_torch.grad.detach().numpy()
    ) < tol
)

BATCH = 5
KERNEL_SIZE = (3, 3)
INPUT_SIZE = (100, 100)
IN_CHANNELS = 32
OUT_CHANNELS = 8 
STRIDE = 3
DILATION = 2
PADDING = 5
GROUPS = 4


X_torch = torch.rand(
        BATCH, IN_CHANNELS, *INPUT_SIZE,
        requires_grad=True
    ) 

X_gg = Tensor(
    X_torch.detach().numpy(),
    requires_grad=True
    )

conv_torch = tnn.Conv2d(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        dilation=DILATION,
        padding=PADDING,
        groups=GROUPS,
        bias=False
    )

conv_gg = gnn.Conv2D(
        out_channels=OUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        dilation=DILATION,
        padding=PADDING,
        groups=GROUPS,
        bias=False
    ) 

conv_gg.w = Tensor(
    conv_torch.weight.detach().numpy(),
    requires_grad=True,
)

out_torch = conv_torch(X_torch)
out_gg = conv_gg(X_gg)

out_torch.sum().backward()
out_gg.sum().backward()

tol = 1e-3
assert np.all(
    abs(out_gg.data 
        - out_torch.detach().numpy()
    ) < tol
)

assert np.all(
    abs(conv_gg.w.grad 
        - conv_torch.weight.grad.detach().numpy()
    ) < tol
)

assert np.all(
    abs(X_gg.grad 
        - X_torch.grad.detach().numpy()
    ) < tol
)

