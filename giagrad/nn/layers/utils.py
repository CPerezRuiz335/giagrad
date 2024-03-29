from itertools import chain, groupby
from typing import Tuple, Optional, List, Callable, Iterable

import numpy as np
from numpy.typing import NDArray
from numpy.lib.stride_tricks import as_strided

"""
Quick notation for some docstrings, e.g. 3D convolution 
with multiple channels with batch training (an image of C_in 
channels now is understood as a 3 dimensional image with Depth,
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

kernel_shape is the shape of the entire filter or kernel.
kernel_size is the shape of each single filter without channels.
conv_dims equals the length of kernel_size.
"""

def flat_tuple(tup: Tuple[Tuple[int, ...] | int, ...]) -> Tuple[int, ...]:
    """flat a tuple made of int or tuples of int"""
    return tuple(chain(*(i if isinstance(i, tuple) else (i,) for i in tup)))

def same_len(*args) -> bool:
    """check if all input parameters have same length"""
    g = groupby(args, lambda x: len(x))
    return next(g, True) and not next(g, False)

def check_parameters(
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...] | int, 
        dilation:  Tuple[int, ...] | int, 
        padding: Tuple[Tuple[int, int] | int, ...] | int | str,
        groups: int,
        out_channels: int
    ): 
    
    if not isinstance(groups, int) or groups <= 0:
        raise ValueError(f"groups must be >= 1 and int, got: {groups}")

    if out_channels // groups == 0 or out_channels % groups != 0:
        raise ValueError(
            f"out_channels={out_channels} are not divisible by groups={groups}"
        )

    stride_, dilation_, padding_ = stride, dilation, padding
    if isinstance(stride, int):
        stride_ = (stride,)*len(kernel_size)

    if isinstance(dilation, int):
        dilation_ = (dilation,)*len(kernel_size)

    if isinstance(padding, int):
        padding_ = (padding,)*len(kernel_size)

    elif padding == 'same':
        padding_ = (0,) * len(kernel_size)

    assert (
        isinstance(stride_, Iterable) 
        and isinstance(dilation_, Iterable) 
        and isinstance(padding_, Iterable)
    ), (
        "stride, dilation and padding must be an iterable or int, got:\n"
        + f"stride: {stride}\n"
        + f"dilation: {dilation}\n"
        + f"padding: {padding}"
    )

    assert same_len(kernel_size, padding_, dilation_, stride_), (
        f"padding, dilation and stride must have the same length as kernel_size"
        + "or be int, got:\n"
        + f"kernel_size: {kernel_size}\n" 
        + f"stride: {stride}\n"
        + f"dilation: {dilation}\n"
        + f"padding: {padding}"
    )
    
    assert len(stride_) == len(kernel_size) and all(
        s >= 1 and isinstance(s, int) for s in flat_tuple(stride_)
    ), f"stride must have positive integers, got: {stride}"

    assert len(dilation_) == len(kernel_size) and all(
        d >= 1 and isinstance(d, int) for d in flat_tuple(dilation_) 
    ), f"dilation must have positive integers, got: {dilation}"

    assert len(padding_) == len(kernel_size) and all(
        p >= 0 and isinstance(p, int) for p in flat_tuple(padding_)
    ), f"padding must have non negative integers, got: {padding}"


def tuple_to_ndarray(fn: Callable):
    def wrapper(*args, **kwargs):
        args = (
            np.array(arg) if isinstance(arg, tuple) else arg for arg in args
        )
        kwargs = {
            k:np.array(v) 
            if isinstance(v, tuple) 
            else v for k, v in kwargs.items()
        }
        return fn(*args, **kwargs)
    return wrapper



@tuple_to_ndarray
def conv_output_shape(
        array_shape: Tuple[int, ...], 
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        dilation: Tuple[int, ...],
        padding: Tuple[Tuple[int, int], ...] # ((Pad_before, Pad_after), ...)
    ) -> NDArray:
    """
    Computes the output shape of a convolution as a generalization 
    of PyTorch's output shape formula for any dimension. For example, 
    in 3d convolution, i.e. (N, C_in, D_in, H_in, W_in), D_out is 
    as follows:

                  | D_in + 2*padding[0] - dilation[0] * (kernel_shape[0] - 1) - 1              |
    D_out = floor | ------------------------------------------------------------- (divide) + 1 |
                  |__                            stride[0]                                   __|
    
    where 
        stride:   (strideD_in, strideH_in, strideW_in)
        dilation: (dilationD_in, ...H_in, ...W_in)
        padding:  (paddingD_in, ...H_in, ...W_in)
    
    NOTE
    ----
    If padding[i] is not symmetric, i.e. is a tuple of the
    amount of padding before and after that dimension, the 
    sum of that tuple is used instead.
    """
    conv_dims = len(kernel_size)
    shape = np.array(array_shape[-conv_dims:])
    padding = 2*padding if padding.ndim == 1 else padding.sum(axis=1)
    return np.floor(
        ((shape + padding - dilation*(kernel_size-1) - 1)) / stride + 1
    ).astype(int)

@tuple_to_ndarray
def padding_same(
        array_shape: Tuple[int, ...],
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        dilation: Tuple[int, ...]
    )-> Tuple[Tuple[int, int]]:
    """
    Computes padding width so that the output convolution has
    same size as the input array. The formula is:
                (shape-1)*stride - shape + kernel_size + (kernel_size-1)*(dilation-1)
    padding =   ---------------------------------------------------------------------
                                                2
    If kernel_size has not equal sizes padding_before and padding_after 
    muest be different, that is why padding for whichever axis is
    (ceil(padding), floor(padding)). This decicision is arbitray,
    changing ceil and floor will lead different results, but the NN will
    learn in both cases.
    """
    conv_dims = len(kernel_size)
    shape = np.array(array_shape[-conv_dims:])
    padding = (
        ((shape-1)*stride - shape + kernel_size + (kernel_size-1)*(dilation-1)) 
        / 2
    ) 

    before = np.ceil(padding).astype(int)
    after = np.floor(padding).astype(int)
    return tuple(tuple(i) for i in np.column_stack((before, after)))


@tuple_to_ndarray
def trans_output_shape(
        array_shape: Tuple[int, ...], 
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        dilation: Tuple[int, ...],
        padding: Tuple[Tuple[int, int], ...] # ((Pad_before, Pad_after), ...)
    ) -> NDArray:
    """
    Compute the output shape of a transposed convolution as a 
    generalization of PyTorch's output shape formula for any dimension. 
    For example, in 3d transposed convolution, i.e. 
    (N, C_in, D_in, H_in, W_in), D_out is as follows:

    D_out = (D_in - 1) * stride[0] - 2*padding[0] 
            + dilation[0] * (kernel_size[0] - 1) + 1
    
    where 
        stride:   (strideD_in, strideH_in, strideW_in)
        dilation: (dilationD_in, ...H_in, ...W_in)
        padding:  (paddingD_in, ...H_in, ...W_in)
    
    NOTE
    ----
    If padding[i] is not symmetric, i.e. is a tuple of the
    amount of padding before and after that dimension, the 
    sum of that tuple is used instead.
    """
    conv_dims = len(kernel_size)
    shape = np.array(array_shape[-conv_dims:])
    return (shape-1)*stride - padding.sum(axis=1) + dilation*(kernel_size-1) + 1

def sliding_filter_view(
        array: NDArray, 
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        dilation: Tuple[int, ...],
    ) -> NDArray:
    """
    Computes as_strided view of the input array in order to 
    make convolutions with tensordot easily. For example, a 
    2D convolution with size (N, C_in, H_in, W_in) with some 
    filter of size (C_out, C_in, kH, kW) multiplies every 
    fiter in C_out H_out * W_out times. 

    Assumes array is batched.

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
            position -8: D_in * H_in * W_in * C_in, next sample (batched)

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
    conv_dims = len(kernel_size)
    in_channels = array.shape[-(conv_dims+1)]
    no_in_channels_shape = array.shape[-conv_dims:]

    # see cumprod pattern in the example: 
    #(W_in * H_in * D_in), (W_in * H_in), ..., 1
    cumprod = np.append(np.cumprod(no_in_channels_shape[::-1])[::-1], (1, ))

    # per-byte strides required to fill and step one window 
    fill_stride = cumprod * np.insert(dilation, 0, values=1)
    step_stride = cumprod * np.insert(stride, 0, values=in_channels) 
    strides = np.append(step_stride, fill_stride)
    output_shape = conv_output_shape(
        array.shape, kernel_size, stride, dilation, padding=((0,0),)*conv_dims
    )
    # shape    
    view_shape = np.concatenate([
        (array.shape[0],), output_shape, (in_channels,), kernel_size
    ])
    # multiply strides by datatype size in bytes
    return as_strided(array, view_shape, strides*array.itemsize)

@tuple_to_ndarray
def dilate(
        array: NDArray, 
        dilation: Tuple[int, ...]
    ) -> NDArray:
    """
    Dilates the input array by params.dilation. This is needed 
    for backpropagation w.r.t input tensor x. 

    The formula for calculating the dilated shape is:

        M = (shape - 1) * (dilation - 1) + shape
    """
    conv_dims = len(dilation)
    shape = np.array(array.shape[-conv_dims:]) # no C_in shape
    M = tuple((shape - 1) * (dilation - 1) + shape)
    dilated_array = np.zeros(array.shape[:2] + M, dtype=array.dtype)

    # create a view with as_strided with the positions that correspond to 
    # the values of the original array and assign those values accordingly 
    strides = dilated_array.strides[:2] 
    strides += tuple(
        x*y for x,y in zip(dilated_array.strides[-conv_dims:], dilation)
    )

    as_strided(dilated_array, array.shape, strides)[:] = array
    return dilated_array

@tuple_to_ndarray
def trimm_uneven_stride(
        array: NDArray, 
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        dilation: Tuple[int, ...]
    ) -> NDArray:
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

        M = (kernel_size - 1) * (dilation - 1) + kernel_size
    
    Therefore, if (sample_size - M) % stride != 0, filter does not
    cover the entire image and it must be sliced.

    NOTE: array is already padded
    """
    conv_dims = len(kernel_size)
    no_in_channels_shape = np.array(array.shape[-conv_dims:])

    M = (kernel_size - 1) * (dilation - 1) + kernel_size 
    offset = (no_in_channels_shape - M) % stride 

    if np.any(offset): 
        slices = (..., ) + tuple(slice(i) for i in no_in_channels_shape-offset)
        return array[slices]
    return array

def convolve_forward(
        x: NDArray,
        w: NDArray,
        stride: Tuple[int, ...],
        dilation: Tuple[int, ...]
    ) -> NDArray:
    """Cross-correlation of input array x with the filter w"""
    conv_dims = w.ndim-2
    kernel_size = w.shape[-conv_dims:]
    # make a view of x ready for tensordot
    sliding_view = sliding_filter_view(x, kernel_size, stride, dilation)
    # convolve the last conv_dims dimensions of w and sliding_view    
    axes = [[-(axis+1) for axis in range(conv_dims+1)],]*2
    conv_out = np.tensordot(w, sliding_view, axes)
    # (C_out, N, W0, ...) -> (N, C_out, W0, ...)
    conv_out = np.swapaxes(conv_out, 0, 1)

    if not conv_out.flags['C_CONTIGUOUS']:
        return np.ascontiguousarray(conv_out)
    return conv_out

def convolve_backward(
        x: NDArray,
        partial: NDArray,
        stride: Tuple[int, ...],
        dilation: Tuple[int, ...]
    ) -> NDArray:
    """
    Cross-correlation of x with partial to obtain weights gradient.
    Unlike convolve forward stride and dilation are swapped and
    tensordot also does reduces by N (batch).
    """
    conv_dims = partial.ndim-2
    kernel_size = partial.shape[-conv_dims:]
    # make a view of x ready for tensordot
    sliding_view = sliding_filter_view(
        x, kernel_size, stride=dilation, dilation=stride # !!!
    )
    # convolve the last conv_dims dimensions of w, 
    # sliding_view, and batch dimension    
    axes = [[0] + [-(axis+1) for axis in range(conv_dims)],]*2
    w_partial = np.tensordot(partial, sliding_view, axes)
    
    # w_partial has shape (N, C_out, X0_out, x1_out, ..., C_in) 
    return np.rollaxis(w_partial, -1, 1) # move C_in to 1st position

def transpose(
        x: NDArray,
        w: NDArray,
        stride: Tuple[int, ...],
        dilation: Tuple[int, ...],
        padding: Optional[Tuple[Tuple[int, int], ...]] = None
    ) -> NDArray:
    """
    Transposed convolution doing forward convolution. x array is dilated
    with stride parameter, and then padded with the size of the dilated
    kernel determined by dilation. It's equivalent to a full convolution
    of a dilated input array by a dilated filter w. Note that this is
    not cross-correlation, that's why w is rotated 180 degrees.

    WARNING
    -------
    Not tested for dimensions > 3 and the implementation is not clear
    for the moment due to determine rotation axes for dimensions > 3.
    """
    conv_dims = w.ndim - 2
    kernel_size = w.shape[-conv_dims:]
    
    # calculate the size of the dilated kernel as in dilate 
    # or trimm_uneven_stride
    M = (np.array(kernel_size) - 1) * (np.array(dilation) - 1) \
        + np.array(kernel_size)

    # padding size equal to dilated_kernel_size-1
    padding = ((0,0),)*2 + tuple((i-1, i-1) for i in M)
    full_x = np.pad(dilate(x, stride), padding)

    # rotate w
    rot_w = np.flip(w, -1) if conv_dims == 1 else np.rot90(w, 2, (-1, -2))

    # windows don't need stride
    stride = (1,)*conv_dims
    sliding_view = sliding_filter_view(full_x, kernel_size, stride, dilation)
    
    # define axes so that (C_out, C_in, KG, KW, ...) -tdot- 
    # (N, C_out, GPad_out, HPad_out, ...) = (C_in, N, G_in, H_in, ...)
    axes = [
        [-(axis+1) for axis in range(conv_dims)] + [0],
        [-(axis+1) for axis in range(conv_dims+1)]
    ]

    trans_out = np.tensordot(rot_w, sliding_view, axes)

    # (C_in, N, G_in, H_in, ...) to (N, C_in, G_in, H_in, ...)
    trans_out = np.swapaxes(trans_out, 0, 1) # move C_in to position 1

    if not trans_out.flags['C_CONTIGUOUS']:
        return np.ascontiguousarray(trans_out)
    return trans_out
