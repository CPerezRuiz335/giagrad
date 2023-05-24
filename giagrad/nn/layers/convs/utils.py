from giagrad.nn.layers.convs.params import ConvParams 
import numpy as np
from numpy.typing import NDArray
from numpy.lib.stride_tricks import as_strided
from typing import Union, Tuple, Optional, Dict, Any, List

def conv_output_shape(array_shape: Tuple[int, ...], params: ConvParams) -> NDArray:
        """
        Compute the output shape of a convolution as a generalization 
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
        padding = params.padding.sum(axis=1)
        shape, dilation = np.array(array_shape[-params.conv_dims:]), params.dilation
        kernel_size, stride = params.kernel_size, params.stride
        ret = np.floor(
            ((shape + padding - dilation*(kernel_size - 1) - 1)) / stride + 1
        ).astype(int)

        return ret

def trans_output_shape(array_shape: Tuple[int, ...], params: ConvParams) -> NDArray:
        """
        Compute the output shape of a transposed convolution as a generalization 
        of PyTorch's output shape formula for any dimension. For example, 
        in 3d transposed convolution, i.e. (N, C_in, D_in, H_in, W_in), D_out is 
        as follows:

        D_out = (D_in - 1) * stride[0] - 2*padding[0] + dilation[0] * (kernel_size[0] - 1) + 1
        
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
        # if one output dimension is negative, it means that 
        # padding is excessive is the only one parameter substracting
        padding = params.padding.sum(axis=1)
        shape, dilation = np.array(array_shape[-params.conv_dims:]), params.dilation
        kernel_size, stride = params.kernel_size, params.stride
        return  (shape - 1) * stride - padding + dilation * (kernel_size - 1) + 1

def sliding_filter_view(array: NDArray, params: ConvParams) -> NDArray:
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
    in_channels = array.shape[-(params.conv_dims+1)]
    no_in_channels_shape = array.shape[-params.conv_dims:]

    # see cumprod pattern in the example: (W_in * H_in * D_in), (W_in * H_in), ..., 1
    cumprod = np.append(np.cumprod(no_in_channels_shape[::-1])[::-1], (1,))
    # per-byte strides required to fill and step one window 
    fill_stride = cumprod * np.insert(params.dilation, 0, values=1)
    step_stride = cumprod * np.insert(params.stride, 0, values=in_channels) 
    strides = np.append(step_stride, fill_stride)
    # shape    
    shape = np.concatenate([
        (array.shape[0],), conv_output_shape(array.shape, params), 
        (in_channels,), params.kernel_size
    ])
    
    # if online learning strides and shape for (N, ...) not needed 
    if params.online_learning:
        shape = shape[1:]
        strides = strides[1:]

    return as_strided(
        array, 
        shape=shape, 
        strides=strides,
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
    no_in_channels_shape = np.array(array.shape[-params.conv_dims:])

    M = (params.kernel_size - 1) * (params.dilation - 1) + params.kernel_size 
    offset = (no_in_channels_shape - M) % params.stride 

    if np.any(offset): 
        slices = (..., ) + tuple(slice(i) for i in no_in_channels_shape-offset)
        return array[slices]
    return array

def trimm_extra_padding(array: NDArray, params: ConvParams) -> NDArray:
    """
    TODO
    """
    P = params.padding - (params.kernel_size - 1) 
    if np.any(P > 0):
        slices = (...,) + tuple(slice(i,-i) if i > 0 else (...,) for i in P)
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
    # create a view with as_strided with the positions that correspond to 
    # the values of the original array and assign those values accordingly 
    as_strided(
        dilated_array, 
        array.shape, 
        np.multiply(array.strides, params.dilation)
    )[:] = array

    return dilated_array