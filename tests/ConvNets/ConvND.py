import sys
sys.path.append('../../')
from giagrad import Tensor
import string
from itertools import chain
from math import floor
import numpy as np
import giagrad.nn as gnn
import giagrad.nn.layers.utils as utils
import torch

torch.manual_seed(123)
# np.random.seed(123)

# from string import (
#     printable
# )

# ##
# print(
#     utils.flat_tuple(
#         ((1,2), 3, (1,4,2,5), (3,))
#     )
# )

# print(
#     utils.same_len(
#             *((1,), (3,), (1,), (3,))
#     )
# )

# print(
#     utils.format_padding(5, 3)
# )

# utils.check_parameters(
#     kernel_size=(3, 2, 2),
#     stride=2,
#     dilation=3,
#     padding=(2,1, 9)
# )

# t = np.empty((12,12), dtype=np.bool_)
# print(t)

# print(
#     utils.trans_output_shape(
#         array_shape=(2, 3, 7, 7),
#         kernel_size=(3, 3),
#         stride=2,
#         dilation=1,
#         padding=((2,2), (2,2))
#     )
# )

# print(
#     utils.trimm_uneven_stride(
#         array=t,
#         kernel_size=(3, 3),
#         stride=2,
#         dilation=1
#     )
# )

# from math import prod

# def create(*args, num=False):
#     N = prod(args)
#     if num:
#         return np.random.randint(
#             0, 4,
#             size=args
#         )
#     return np.array(
#         list((printable+printable)[:N]),
#         dtype=object
#     ).reshape(args)

# t = create(2, 2, 3)
# t = np.pad(
#     t, 
#     pad_width=(
#         ((0,0), (0,0), (2,2))
#     ),
#     constant_values='0'
# )
# print('# t\n', t)

# k = create(2, 2, 2, num=True)
# print('# k\n', k)


import torch.nn as tnn

X_torch = torch.rand(
        2, 3, 5, 5,
        requires_grad=True
    ) 
X_gg = Tensor(
    X_torch.detach().numpy(),
    requires_grad=True
    )

conv_torch = tnn.Conv2d(
        in_channels=3,
        out_channels=4,
        kernel_size=(3, 3),
        stride=1,
        dilation=1,
        bias=False
    )

conv_gg = gnn.Conv2D(
        out_channels=4,
        kernel_size=(3,3),
        stride=1,
        dilation=1,
        bias=False
    ) 

conv_gg.w = Tensor(
    conv_torch.weight.detach().numpy(),
    requires_grad=True,
)
out_torch = conv_torch(X_torch)
out_gg = conv_gg(X_gg)
print(out_torch)
print(out_gg)

tol = 1e-6
assert np.all(
    abs(out_gg.data 
        - out_torch.detach().numpy()
    ) < tol
)

out_torch.sum().backward()
print('kernl', conv_gg.w.shape)
print('out size', out_gg.shape)
print('out size', out_torch.shape)
out_gg.sum().backward()

print(conv_torch.weight.grad)
print(conv_gg.w.grad)

exit()



print('convolved shape:', 
(convolved :=
    utils.convolve(
        t, k,
        stride=(1,),
        dilation=(2,)
    )
).shape
)

print('# convolved\n', convolved)

convolved = create(
    *convolved.shape, num=True
)

print('# convolved\n', convolved)

print(
(transposed :=
    utils.transpose(
        convolved, k,
        stride=(1,),
        dilation=(2,),
        padding=((2,2),)
    )
).shape
)

print('# transposed\n', transposed)


t = Tensor.empty(
    2, 2, 3,
    requires_grad=True,
    dtype=np.float32
).uniform(-10, 10)

out = conv1d(t)
print('t\n', t)
print('kernel\n', conv1d.w)
print('out\n', out)
out.sum().backward()
print(conv1d.w.grad)
print(t.grad)