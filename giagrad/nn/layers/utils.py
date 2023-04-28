# type: ignore
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Iterable, Callable
from numpy.lib.stride_tricks import as_strided

"""
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
            image
    kH: kernel height 
    kW: kernel widht

Likewise, for higher order convolutions such as 3D ones, every filter
has also kD (filter depth).  
"""

def in_tuple_to_numpy(fn: Callable):
    def wrapper(*args, **kwargs):
        # to numpy array input
        return fn(
            *(np.array(arg) if isinstance(arg, tuple) else arg for arg in args), 
            **{key: np.array(value) 
                if isinstance(value, tuple) else value 
                for key, value in kwargs.items()
            }
        )
    return wrapper

@in_tuple_to_numpy
def _output_shape(
        array_shape: Tuple[int],
        kernel_shape: Tuple[int], 
        stride: Tuple[int], 
        dilation: Tuple[int],
        padding: Tuple[int]
    ) -> Tuple[int]:
    """
    Compute output shape as a generalization of PyTorch output shape 
    formula for any dimension. For example, in 3d convolution, i.e.
    (N, C_in, D_in, H_in, W_in), D_out is as follows:

                  | D_in + 2*padding[0] - dilation[0] * (kernel_shape[0] - 1) - 1               |
    D_out = floor | ------------------------------------------------------------- (divide) + 1  |
                  |                             stride[0]                                       |

    where 
        stride:   (strideD_in, strideH_in, strieW_in)
        dilation: (dilationD_in, ...H_in, ...W_in)
        padding:  (paddingD_in, ...H_in, ...W_in)
    """
    conv_dims = len(kernel_shape) - 2
    batch_or_none = array_shape[:-(conv_dims+1)]
    C_out = kernel_shape[0]
    
    ret = np.floor(
        (array_shape[-conv_dims:] + 2*padding - dilation*(kernel_shape[-conv_dims:] - 1) - 1) \
        / stride \
        + 1
    ).astype(np.int32)
    
    # return tuple(np.concatenate((batch_or_none, (C_out,), ret)))
    return tuple(ret)

def _kernel_ready_as_strided(
        array: NDArray, 
        kernel_shape: Tuple[int], 
        stride: Tuple[int], 
        dilation: Tuple[int],
        output_shape: Tuple[int],
        online_learning: bool = False
    ) -> NDArray:
    """
    Computes as_strided view of the input array in order to make 
    convolutions with tensordot easily. For example, a 2D convolution 
    with size (N, C_in, H_in, W_in) with some filter of size 
    (C_out, C_in, kH, kW) multiplies every fiter in C_out 
    H_out * W_out times. 

    The strided view will have dimensions (N, H_out, W_out, C_in, kH, kW),
    so every kernel will be multiplied element wise for every (..., H_out, W_out)
    compatible view and reduced using sum (this is done by tensordot). 
    Note that the last three axes of (N, H_out, W_out, C_in, kH, kW) are compatible 
    with the three of (C_out, C_in, kH, kW).

    How to calculate view shape and strides for 3D convolution:

    - strides
        [ -8, -7, -6 , -5 , -4 , -3 , -2 , -1 ] * itemsize in bytes
        
        position -1: dilation[2], horizontal dilation
        position -2: W_in * dilation[1], vertical dilation
        position -3: W_in * H_in * dilation[0], depth dilation
        position -4: W_in * H_in * D_in, jump to next Channel
        position -5: stride[2], horizontal stride
        position -6: W_in * stride[1], vertical stride
        position -7: W_in * H_in * stride[0], depth stride
        position -8: W_in * H_in * D_in * C_in, next observation in Batched data

    - shape
        [ -8, -7, -6 , -5 , -4 , -3 , -2 , -1 ]
        
        position -1: kW
        position -2: kH
        position -3: kD
        position -4: Cin
        position -5: Wout
        position -6: Hout
        position -7: Dout
        position -8: N
        
    Note
    ----
        Online learning means first axis is C_in, because data is not batched.
    """
    assert array.flags["C_CONTIGUOUS"], "_kernel_ready_as_strided only accepts contiguous arrays"

    # precompute some values
    sample_dims = array.ndim - (0 if online_learning else 1)
    num_channels = array.shape[-sample_dims]
    no_channels_shape = array.shape[-sample_dims+1:]

    # compute decreasing factorial like pattern as in the example
    fact_pattern = [np.prod(no_channels_shape[i:], dtype=np.int32) for i in range(sample_dims)] * 2
    remaining_pattern = (num_channels,) + stride + (1,) + dilation
    
    # calculate strides and the as_strided output shape, if online learning N value not needed
    strides = [a*b*array.itemsize for a, b in zip(fact_pattern, remaining_pattern)][online_learning:]
    shape = ((array.shape[0],) + output_shape + kernel_shape[1:])[online_learning:]
    print('strides no byte: ', np.array(strides)/array.itemsize)
    print('strides in bytes: ', strides)
    print('strided output shape: ', shape)
    
    return as_strided(array, shape=shape, strides=strides)


@in_tuple_to_numpy
def _trimm_uneven_stride(
        array: NDArray, # already padded
        kernel_shape: Tuple[int],
        stride: Tuple[int],
        dilation: Tuple[int],
        online_learning: bool = False
    ) -> NDArray:
    """
    If the strides are not carefully chosen, the filter may 
    not cover the entire image, leaving out the last columns 
    or rows. In this case, these columns or rows need to be sliced.

    This step is necessary for backpropagation, especially when the 
    strides are not appropriately chosen. However, it comes at the 
    cost of losing some information from the input data.
    
    A dilated filter spans an M_i number of entries in any direction
    _i where M equals:

        M = (kernel_size - 1) * (dilation - 1) + kernel_shape
    
    Therefore, if (sample_size - M) % stride != 0, filter does not
    cover the entire image and it must be sliced.
    """
    # precompute some values
    sample_dims = array.ndim - (0 if online_learning else 1)
    no_channels_shape = array.shape[-sample_dims+1:]
    kernel_size = kernel_shape[2:]

    M = (kernel_size - 1) * (dilation - 1) + kernel_size 
    offset = (no_channels_shape - M) % stride 

    if np.any(offset):
        slices = (..., ) + tuple(slice(i) for i in no_channels_shape-offset)
        return array[slices]

    return array