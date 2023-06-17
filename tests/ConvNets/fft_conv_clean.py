import sys
from functools import partial
from typing import Iterable, Tuple, Union
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import numpy as np
from numpy.typing import NDArray

sys.path.append('../../')
from giagrad.nn.layers.utils import convolve_forward


def complex_matmul(a: NDArray, b: NDArray, groups: int = 1) -> NDArray:
    a = a.reshape((a.shape[0], groups, -1, *a.shape[2:]))
    b = b.reshape((groups, -1, *b.shape[1:]))

    a = np.expand_dims(np.moveaxis(a, 2, -1), -2)
    b = np.moveaxis(b, (1, 2), (-1, -2))

    # complex value matrix multiplication
    real = np.real(a) @ np.real(b) - np.imag(a) @ np.imag(b)
    imag = np.imag(a) @ np.real(b) + np.real(a) @ np.imag(b)
    real = np.squeeze(np.moveaxis(real, -1, 2), -1)
    imag = np.squeeze(np.moveaxis(imag, -1, 2), -1)
    c = np.zeros(real.shape, dtype=np.complex64)
    c.real = real 
    c.imag = imag

    return c.reshape(c.shape[0], -1, *c.shape[3:])


def fft_conv(
    signal: NDArray,
    kernel: NDArray,
    padding: Tuple[int],
    stride: Tuple[int],
    dilation: Tuple[int],
    padding_mode: str = "constant",
    groups: int = 1,
) -> NDArray:
    # Cast padding, stride & dilation to tuples.
    n = signal.ndim - 2
    padding_ = padding
    stride_ = stride
    dilation_ = dilation

    # internal dilation offsets
    offset = np.zeros((1, 1) + dilation_, dtype=np.float64)
    offset[(slice(None), slice(None), *((0,) * n))] = 1.0

    # correct the kernel by cutting off unwanted dilation trailing zeros
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)

    # pad the kernel internally according to the dilation parameters
    kernel = np.kron(kernel, offset)[(...,) + cutoff]

    # Pad the input signal & kernel tensors
    signal_padding = [(0,0)]*2 + [(p,p) for p in padding_]
    signal = np.pad(signal, signal_padding)

    if signal.shape[-1] % 2 != 0:
        signal_ = np.pad(signal, (0,0)*(signal.ndim-1) + (0, 1))
    else:
        signal_ = signal

    kernel_padding = [(0,0)]*2 + [
        tuple(pad for pad in [0, signal_.shape[i] - kernel.shape[i]])
        for i in range(2, signal_.ndim)
    ]

    padded_kernel = np.pad(kernel, kernel_padding)

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    # signal_ = signal_.reshape(signal_.size(0), groups, -1, *signal_.shape[2:])
    signal_fr = np.fft.rfftn(signal_, axes=tuple(range(2, signal.ndim)))
    kernel_fr = np.fft.rfftn(padded_kernel, axes=tuple(range(2, signal.ndim)))

    kimag = np.imag(kernel_fr) 
    kimag *= -1

    output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    output = np.fft.irfftn(output_fr, axes=tuple(range(2, signal.ndim)))

    # Remove extra padded values
    crop_slices = (...,)  + tuple(
        slice(0, (signal.shape[i] - kernel.shape[i] + 1), stride_[i - 2])
        for i in range(2, signal.ndim)
    )

    output = np.ascontiguousarray(output[crop_slices]) 

    return output

if __name__ == "__main__":
    np.random.seed(1234)
    BATCH = 1
    IN_CHANNELS = 2
    OUT_CHANNELS = 2
    GROUPS = 1

    zeros = lambda n: (0,)*n
    ones = lambda n: (1,)*n
    signal_shape = lambda s_size: (BATCH,) + (IN_CHANNELS,) + s_size
    kernel_shape = lambda k_size: (OUT_CHANNELS,) + (IN_CHANNELS,) + k_size


    funcs = [
        # ('tensordot', convolve_forward),  
        ('fft', fft_conv)
    ]
    input_sizes = list(map(
        signal_shape, 
        [(4096,)]*3 + [(512, 512)]*3 + [(64, 64, 64)]*3
    ))
    kernel_sizes = list(map(
        kernel_shape, 
        [
            (2,), (50,), (500,), 
            (2, 2), (100, 100), (300, 300),
            (2, 2, 2), (15, 15, 15), (60, 60, 60)
        ]
    ))

    results = {
        'function': [],
        'time': [],
        'input_size': [],
        'kernel_size': []
    }

    for name, func in funcs:
        for signal, kernel in zip(input_sizes, kernel_sizes):
            signal_data = np.random.rand(*signal)
            kernel_data = np.random.rand(*kernel)
            print(name)
            print(f"{signal = }")
            print(f"{kernel = }")

            if name == 'tensordot':
                t1 = timer()
                func(
                    x=signal_data,
                    w=kernel_data,
                    stride=ones(len(kernel[2:])),
                    dilation=ones(len(kernel[2:]))
                )
                t2 = timer()

            else:
                t1 = timer()
                func(
                    signal_data,
                    kernel_data,
                    padding=zeros(len(kernel[2:])),
                    stride=ones(len(kernel[2:])),
                    dilation=ones(len(kernel[2:]))
                )
                t2 = timer()
            
            results['function'].append(name)
            results['time'].append(t2-t1)
            results['input_size'].append(np.prod(signal))
            results['kernel_size'].append(np.prod(kernel))

    data = pd.DataFrame(results)
    g = sns.relplot(
        data=data,
        x="input_size", y="kernel_size",
        hue="time", size="time"
    )
    plt.show()
