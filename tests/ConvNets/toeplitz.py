import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy as sp

import sys
from sys import getsizeof
sys.path.append('../../')

from giagrad.nn.layers.utils import conv_output_shape
import string
from itertools import chain

letters = [chr(i) for i in chain(range(33, 126), range(180, 1200))]

## define shapes

BATCH = 1
IN_CHANNELS = 16
INPUT_SIZE = np.array((28, 28))
KERNEL_SIZE = np.array((2, 2))
OUT_CHANNELS = 16

STRIDE = np.array((1,1))
DILATION = np.array((1,1))
PADDING = (0,0) # ALWAYS ASSUME ALREADY PADDED ARRAY

INPUT_SHAPE = np.concatenate([(BATCH, IN_CHANNELS), INPUT_SIZE])
KERNEL_SHAPE = np.concatenate([(OUT_CHANNELS, IN_CHANNELS), KERNEL_SIZE])

OUTPUT_SIZE = conv_output_shape(
    INPUT_SHAPE, KERNEL_SIZE, STRIDE, DILATION, PADDING
)

OUTPUT_SHAPE = np.array((BATCH, OUT_CHANNELS) + tuple(OUTPUT_SIZE))

TOEPLITZ_SHAPE = np.array((np.prod(OUTPUT_SIZE)*OUT_CHANNELS, np.prod(INPUT_SIZE)*IN_CHANNELS))

DILATED_KERNEL_SIZE = (KERNEL_SIZE - 1) * (DILATION - 1) + KERNEL_SIZE

def print_dimensions():
    print(f"{INPUT_SHAPE = }")
    print(f"{KERNEL_SHAPE = }")
    print(f"{OUTPUT_SIZE = }")
    print(f"{TOEPLITZ_SHAPE = }")
    print(f"{OUTPUT_SHAPE = }")
    print(f"{DILATED_KERNEL_SIZE = }")
    
print_dimensions()

# init array
np.random.seed(123)
# X = np.array(letters[:np.prod(INPUT_SHAPE)], dtype=object).reshape(INPUT_SHAPE)
X = np.random.rand(*INPUT_SHAPE)
W = np.random.rand(*KERNEL_SHAPE)
T = np.zeros(TOEPLITZ_SHAPE)  # toeplitz

def print_arrays():
    print('X', X, 'W', W, 'T', T, sep='\n')
# print_arrays()

# window view o Toeplitz matrix

TH, TW = T.shape
H_in, W_in = INPUT_SIZE
KH, KW = KERNEL_SIZE
H_out, W_out = OUTPUT_SIZE
stride_H, stride_W = STRIDE
dilation_H, dilation_W = DILATION
KDil_H, KDil_W = DILATED_KERNEL_SIZE

STRIDED_SHAPE = np.concatenate([OUTPUT_SIZE, (OUT_CHANNELS, IN_CHANNELS), KERNEL_SIZE])
STRIDES = (
    TW * W_out + W_in*stride_H,
    TW + stride_W,
    TW * H_out*W_out, # out channels, jump rows in toeplitz
    H_in*W_in, # jumpt to other channel
    (W_in-KDil_W) + W_in*(dilation_H-1) + KDil_W,
    dilation_W
)

print(f"{STRIDED_SHAPE = }")
print(f"{STRIDES = }")

# as_strided view
WINDOW_VIEW = as_strided(
    T, 
    STRIDED_SHAPE, 
    np.array(STRIDES)*T.itemsize)

print(f"{WINDOW_VIEW.shape = }")
print(f"{W.shape = }")

WINDOW_VIEW += W

import numba as nb
from typing import *
from numpy.typing import NDArray
from time import time 


# def test(t, x, out_shape, n) -> None:
#     for _ in range(n):
#         (t @ x) 

# t1 = time()
# WINDOW_VIEW = as_strided(
#     T, 
#     STRIDED_SHAPE, 
#     np.array(STRIDES)*T.itemsize)

# WINDOW_VIEW += W
# test(np.repeat(T, 2**8, axis=0), X.reshape(-1, 1), OUTPUT_SHAPE, n=1)
# print(f"toeplitz: {time()-t1}")


## torch

from giagrad import Tensor
import numpy as np
import giagrad.nn as gnn
import torch
import torch.nn as tnn

X_torch = torch.rand(
        2**8, 32, *INPUT_SIZE,
        requires_grad=True
    ) 

X_gg = Tensor(
    X_torch.detach().numpy().copy(),
    requires_grad=True
    )

conv_torch = tnn.Conv2d(
        in_channels=32,
        out_channels=32,
        kernel_size=tuple(int(i) for i in KERNEL_SIZE),
        stride=tuple(int(i) for i in STRIDE),
        dilation=tuple(int(i) for i in DILATION),
        bias=False
    )

conv_gg = gnn.Conv2D(
        out_channels=32,
        kernel_size=tuple(int(i) for i in KERNEL_SIZE),
        stride=tuple(int(i) for i in STRIDE),
        dilation=tuple(int(i) for i in DILATION),
        bias=False
    ) 

conv_gg.w = Tensor(
    conv_torch.weight.detach().numpy().copy(),
    requires_grad=True,
)

t1 = time()
out_torch = conv_torch(X_torch)
print(out_torch.shape)
out_torch.sum().backward(retain_graph=True)
print(f"pytorch: {time()-t1}")

print("\n#giargad")

t1 = time()
out_gg = conv_gg(X_gg)
out_gg.sum().backward()
print(f"giagrad: {time()-t1}")
