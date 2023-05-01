from __future__ import annotations
import numpy as np
from itertools import groupby
from numpy.typing import NDArray
from typing import Tuple, Iterable, Callable, Union, Dict, Any, Optional
from numpy.lib.stride_tricks import as_strided
from itertools import chain
from dataclasses import dataclass, asdict
from giagrad import Tensor

"""
NOTE
----
Quick notation, e.g. 3D convolution with multiple 
channels with batch trainig (an image of C_in channels 
now is understood as a 3 dimensional image with Depth,
Height and Width):

    N: number of observations in batch
    C_in: number of input channels
    D_in: input depth size
    H_in: input height size
    W_in: input width size

For higher dimensions same pattern applies, let's say
fourth dimension is G_in:

    N: number of observations in batch
    C_in: number of input channels
    G_in: input number of 3D arrays of size (D_in, H_in, W_in)
    D_in: input depth size
    H_in: input height size
    W_in: input width size

For kernels, the dimensions are:
    
    C_out: number of channels each observations has after applying 
            that convolution, i.e. number of kernels/filters to 
            be applied.
    C_in: every kernel/filter has the same channels as the input 
            image (if no grouped convolution).
    kH: kernel height 
    kW: kernel widht

Likewise, for higher order convolutions such as 3D ones, every filter
has also kD (filter depth).  

kernel_shape is the shape of the entire filter or kernel
kernel_size is the shape of each single filter without channels 

WARNING
-------
For simplicity, every helper function assumes input data is batched,
so if online_learning is done just expand 0 dimension. This does not 
cost much, I would say nothing as it does not add data, just changes 
strides, and it is also a view, not a new array.
"""

# utils of utils
def format_padding(
        padding: Tuple[Union[Tuple[int, ...], int], ...]
    ) -> Tuple[Tuple[int, ...], ...]:
    """
    If an iterator is supplied to numpy.pad's pad_width parameter
    it must have the same length as the array that is going to be 
    padded and every element must be of the same type. 

    For a 2D array, pad_width = (1, (3, 2)) not accepted,
    pad_width = ((1,), (3, 2)) not accepted, pad_width =
    ((1,1), (3,1)) accepted. 
    """
    return tuple(i if isinstance(i, tuple) else (i, i) for i in padding)

def flat_tuple(tup: Tuple[Union[Tuple[int, ...], int], ...]) -> Tuple[int, ...]:
    """flat a tuple made of int or tuples of int"""
    return tuple(chain(*(i if isinstance(i, tuple) else (i,) for i in tup)))

def same_len(*args: Tuple[int, ...]):
    """check if all input tuples have same length"""
    g = groupby(args, lambda x: len(x))
    return next(g, True) and not next(g, False)

# main utils
class ConvParams:
    """
    Input paremters that define convolution must be well
    formated and defined because util functions work with
    numpy arrays not tuples, and as_strided can cause
    segmentation faults, for example.

    This also simplifies communication of parameters between 
    ConvND(Function) and every layer(Module), and gives the 
    responsability of checking the input parameter's format
    to ConvParams.

    NOTE: Room for improvement is welcome.
    """
    def __init__(
        self,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Union[Tuple[int, ...], int],
        dilation:  Union[Tuple[int, ...], int],
        padding: Union[Tuple[Union[Tuple[int, ...], int], ...], int],
        padding_mode: str,
        padding_kwargs: Dict[str, Any],
        groups: int,
        bias: bool,
        # other
        online_learning: bool = False,
        is_copy: bool = False,
        axis_pad: Optional[NDArray] = None
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.padding_kwargs = padding_kwargs
        self.groups = groups
        self.bias = bias
        self.out_channels = out_channels
        self.conv_dims = len(kernel_size)
        self.online_learning = online_learning
        self.is_copy = is_copy
        self.axis_pad = axis_pad

        if not self.is_copy: 
            self._init()

    def _init(self):
        if isinstance(self.stride, int):
            self.stride = (self.stride,)*len(self.kernel_size)

        if isinstance(self.dilation, int):
            self.dilation = (self.dilation,)*len(self.kernel_size)

        if isinstance(self.padding, int):
            self.padding = (self.padding,)*len(self.kernel_size)

        if not (
            isinstance(self.padding, tuple) 
            and isinstance(self.padding, tuple) 
            and isinstance(self.padding, tuple)
        ):
            msg = "stride, dilation and padding must be a tuple or int, got:\n"
            msg += f"stride: {self.stride}"
            msg += f"dilation: {self.dilation}"
            msg += f"padding: {self.padding}"
            raise ValueError(msg)

        assert same_len(self.kernel_size, self.padding, self.dilation, self.stride),\
        f"kernel_size, padding, dilation and stride must have the same length"
        
        assert len(self.stride) == len(self.kernel_size) and all(
            s >= 1 and isinstance(s, int) for s in flat_tuple(self.stride)
        ), f"stride must have positive integers, got: {self.stride}"

        assert len(self.dilation) == len(self.kernel_size) and all(
            d >= 1 and isinstance(d, int) for d in flat_tuple(self.dilation) 
        ), f"dilation must have positive integers, got: {self.dilation}"

        assert len(self.padding) == len(self.kernel_size) and all(
            p >= 0 and isinstance(p, int) for p in flat_tuple(self.padding)
        ), f"padding must have non negative integers, got: {self.padding}"

        self.padding = format_padding(self.padding)

        # make every tuple numpy array
        for key, value in vars(self).items():
            if isinstance(value, tuple):
                setattr(self, key, np.array(value))

    def set_axis_pad(self, x: Tensor):
        self.axis_pad = np.append(
            ((0,0),)*(x.ndim - self.conv_dims), 
            self.padding, 
            axis=1
        )

    def copy(self) -> ConvParams:
        dd = vars(self).copy()
        dd.update({'copy': True})
        return ConvParams(**dd)

    def output_shape(self, array_shape: Tuple[int, ...]) -> NDArray:
        """
        Compute output shape as a generalization of PyTorch 
        output shape formula for any dimension. For example, 
        in 3d convolution, i.e. (N, C_in, D_in, H_in, W_in), 
        D_out is as follows:

                      | D_in + 2*padding[0] - dilation[0] * (kernel_shape[0] - 1) - 1              |
        D_out = floor | ------------------------------------------------------------- (divide) + 1 |
                      |__                            stride[0]                                   __|
        
        If padding[i] is not symmetric, i.e. is a tuple of the
        amount of padding before and after that dimension, the 
        sum of that tuple is used instead.

        where 
            stride:   (strideD_in, strideH_in, strieW_in)
            dilation: (dilationD_in, ...H_in, ...W_in)
            padding:  (paddingD_in, ...H_in, ...W_in)
        """
        padding = self.padding.sum(axis=1)
        shape, dilation = np.array(array_shape[-self.conv_dims:]), self.dilation
        kernel_size, stride = self.kernel_size, self.stride
        return np.floor(
            ((shape + padding - dilation*(kernel_size - 1) - 1)) / stride + 1
        ).astype(int)

def kernel_ready_view(array: NDArray, params: ConvParams) -> NDArray:
    """
    Computes as_strided view of the input array in order to 
    make convolutions with tensordot easily. For example, a 
    2D convolution with size (N, C_in, H_in, W_in) with some 
    filter of size (C_out, C_in, kH, kW) multiplies every 
    fiter in C_out H_out * W_out times. 

    The strided view will have dimensions (N, H_out, W_out, C_in, kH, kW), 
    so every kernel will be multiplied elementwise for every 
    (..., H_out, W_out, ...) compatible view and reduced using 
    sum (this is done by tensordot). Note that the last three 
    axes of (N, H_out, W_out, C_in, kH, kW) are compatible with 
    the three of (C_out, C_in, kH, kW).

    How to calculate as_strided's shape and strides for 3D convolution 
    with shape (N, C_in, D_in, H_in, W_in), from innermost dimensions 
    to outermost ones:

    - strides:

        [ -8, -7, -6 , -5 , -4 , -3 , -2 , -1 ] * itemsize in bytes
        
        strides to fill a window:
            position -1: 1 * dilation[2], horizontal dilation
            position -2: W_in * dilation[1], vertical dilation
            position -3: H_in * W_in * dilation[0], depth dilation
            position -4: D_in * H_in * W_in * 1, jump to next Channel

        strides to step to the next window:
            position -5: 1 * stride[2], horizontal stride
            position -6: W_in * stride[1], vertical stride
            position -7: H_in * W_in * stride[0], depth stride
            position -8: D_in * H_in * W_in * C_in, next observation in batched data

    - shape:

        [ -8, -7, -6 , -5 , -4 , -3 , -2 , -1 ]
        
        position -1: kW
        position -2: kH
        position -3: kD
        position -4: Cin
        position -5: Wout
        position -6: Hout
        position -7: Dout
        position -8: N
    """
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)

    # precompute some values
    in_channels, no_in_channels_shape = array.shape[1], array.shape[2:]

    # see cumprod pattern in the example: (W_in * H_in * D_in), (W_in * H_in), ..., 1
    cumprod = np.append(np.cumprod(no_in_channels_shape[::-1])[::-1], (1,))
    # per-byte strides required to fill and step one window 
    fill_stride = cumprod * np.insert(params.dilation, 0, values=1)
    step_stride = cumprod * np.insert(params.stride, 0, values=in_channels) 
    # shape    
    shape = np.concatenate([
        (array.shape[0],), params.output_shape(array.shape), 
        (in_channels,), params.kernel_size
    ])
    
    return as_strided(
        array, 
        shape=shape, 
        strides=np.append(step_stride, fill_stride),
        writeable=False
    )

def trimm_uneven_stride(array: NDArray, params: ConvParams) -> NDArray:
    """
    If the strides are not carefully chosen, the filter may 
    not cover the entire image, leaving out the last columns 
    or rows. In this case, these columns or rows need to be sliced.

    This step is necessary for backpropagation, especially when the 
    strides are not appropriately chosen. However, it comes at the 
    cost of losing some information from the input data, but that is 
    the user's responsability.
    
    A dilated filter spans an M_i number of entries in any direction
    _i where M equals:

        M = (kernel_size - 1) * (dilation - 1) + kernel_shape
    
    Therefore, if (sample_size - M) % stride != 0, filter does not
    cover the entire image and it must be sliced.
    """
    # pad first
    if np.sum(params.padding).item():
            array = np.pad(
                array, params.axis_pad, 
                mode=params.padding_mode, 
                **params.padding_kwargs
            ) 

    # precompute some values
    no_in_channels_shape = np.array(array.shape[2:])

    M = (params.kernel_size - 1) * (params.dilation - 1) + params.kernel_size 
    offset = (no_in_channels_shape - M) % params.stride 

    if np.any(offset): 
        slices = (..., ) + tuple(slice(i) for i in no_in_channels_shape-offset)
        return array[slices]
    return array

def dilate(array: NDArray, params: ConvParams):
    """
    Dilates the input array by params.dilation. This is needed 
    for backpropagation w.r.t input tensor x. 

    The formula for calculating the dilated shape is:

        M = (shape - 1) * (dilation - 1) + shape
    """
    shape = np.array(array.shape)
    M = (shape - 1) * (params.dilation - 1) + shape
    dilated_array = np.zeros(M, dtype=array.dtype)
    # create a view with as_strided of the positions that correspond to 
    # the values of the original array and assign those values accordingly 
    as_strided(
        dilated_array, 
        array.shape, 
        np.multiply(array.strides, params.dilation)
    )[:] = array

    return dilated_array