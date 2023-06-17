from functools import partial
from typing import Iterable, Tuple, Union

import sys
sys.path.append('../../')
from giagrad.nn.layers.utils import conv_output_shape


import torch
import torch.nn.functional as f
from torch import Tensor, nn
from torch.fft import irfftn, rfftn
import numpy as np


def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    print("\n\n### COMPLEX MATMUL ###")
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    print(f"{a.shape = }")
    print(f"{b.shape = }")

    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))

    print(f"{a.shape = }")
    print(f"{b.shape = }")


    # complex value matrix multiplication
    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])

def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """Casts to a tuple with length 'n'.  Useful for automatically computing the
    padding and stride for convolutions, where users may only provide an integer.

    Args:
        val: (Union[int, Iterable[int]]) Value to cast into a tuple.
        n: (int) Desired length of the tuple

    Returns:
        (Tuple[int, ...]) Tuple of length 'n'
    """
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)


def fft_conv(
    signal: Tensor,
    kernel: Tensor,
    bias: Tensor = None,
    padding: Union[int, Iterable[int]] = 0,
    padding_mode: str = "constant",
    stride: Union[int, Iterable[int]] = 1,
    dilation: Union[int, Iterable[int]] = 1,
    groups: int = 1,
) -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).

    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Tensor) Bias tensor to add to the output.
        padding: (Union[int, Iterable[int]) Number of zero samples to pad the
            input on the last dimension.
        stride: (Union[int, Iterable[int]) Stride size for computing output values.

    Returns:
        (Tensor) Convolved tensor
    """

    # Cast padding, stride & dilation to tuples.
    n = signal.ndim - 2
    padding_ = to_ntuple(padding, n=n)
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n)

    # internal dilation offsets
    offset = torch.zeros(1, 1, *dilation_, device=signal.device, dtype=signal.dtype)
    offset[(slice(None), slice(None), *((0,) * n))] = 1.0

    # correct the kernel by cutting off unwanted dilation trailing zeros
    cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)

    # pad the kernel internally according to the dilation parameters
    kernel = torch.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]
    print(f"dilated kernel = {kernel}")
    # Pad the input signal & kernel tensors
    signal_padding = [p for p in padding_[::-1] for _ in range(2)]
    signal = f.pad(signal, signal_padding, mode=padding_mode)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    if signal.size(-1) % 2 != 0:
        signal_ = f.pad(signal, [0, 1])
    else:
        signal_ = signal

    print(f"check if signal padding has even size\n{signal_ = }")

    kernel_padding = [
        pad
        for i in reversed(range(2, signal_.ndim))
        for pad in [0, signal_.size(i) - kernel.size(i)]
    ]
    padded_kernel = f.pad(kernel, kernel_padding)

    print(f"{padded_kernel = }")

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    # signal_ = signal_.reshape(signal_.size(0), groups, -1, *signal_.shape[2:])
    signal_fr = rfftn(signal_, dim=tuple(range(2, signal.ndim)))
    kernel_fr = rfftn(padded_kernel, dim=tuple(range(2, signal.ndim)))

    print(f"{signal.shape = }")
    print(f"{kernel.shape = }")
    print(f"{signal_fr.shape = }")
    print(f"{kernel_fr.shape = }")

    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)

    print(f"{output_fr.shape = }")
    output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))
    print(f"{output.shape = }")

    # Remove extra padded values
    crop_slices = [slice(0, output.size(0)), slice(0, output.size(1))] + [
        slice(0, (signal.size(i) - kernel.size(i) + 1), stride_[i - 2])
        for i in range(2, signal.ndim)
    ]
    output = output[crop_slices].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output



if __name__ == "__main__":
	np.random.seed(1234)
	BATCH = 1
	KERNEL_SIZE = (2, 2)
	INPUT_SIZE = (5, 5)
	IN_CHANNELS = 2 
	OUT_CHANNELS = 2
	STRIDE = (1, 1)
	DILATION = (1, 1)
	PADDING = (1, 3)
	GROUPS = 1

	SIGNAL_SHAPE = (BATCH,) + (IN_CHANNELS,) + INPUT_SIZE
	KERNEL_SHAPE = (OUT_CHANNELS,) + (IN_CHANNELS,) + (KERNEL_SIZE)

	t = np.random.randint(4, size=SIGNAL_SHAPE).astype(np.float32)
	k = np.random.randint(2, size=KERNEL_SHAPE).astype(np.float32)
	t_torch = torch.from_numpy(t)
	k_torch = torch.from_numpy(k)

	print(f"{t_torch = }")
	print(f"{k_torch = }")

	output = fft_conv(
	    signal = t_torch,
	    kernel = k_torch,
	    bias = None,
	    padding = PADDING,
	    padding_mode = "constant",
	    stride = STRIDE,
	    dilation = DILATION,
	    groups = GROUPS,
	)

	output_shape = conv_output_shape(
	    t.shape,
	    KERNEL_SIZE,
	    STRIDE, 
	    DILATION,
	    PADDING
	)
	print("\n\n### OUTPUT ###")
	print(f"output shape {output_shape}")
	print(f"{output.shape = }")
	print(f"{output = }")