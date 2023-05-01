import numpy as np
from itertools import groupby
from numpy.typing import NDArray
from typing import Tuple, Iterable, Callable, Union
from numpy.lib.stride_tricks import as_strided
from itertools import chain

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
            image (if no grouped convolution).
    kH: kernel height 
    kW: kernel widht

Likewise, for higher order convolutions such as 3D ones, every filter
has also kD (filter depth).  

WARNING
-------
For simplicity, every helper function assumes input data is batched,
so if online_learning is done just expand 0 dimension. This does not 
cost much, I would say nothing as it does not add data, just changes 
strides, and it is also a view, not a new array.

NOTE
----
kernel_shape is the shape of the entire filter or kernel
kernel_size is the shape of each single filter without channels 
"""

# utils of utils, for the sake of it
def same_len(*args: Tuple[int, ...]):
    """check if all input tuples have same length"""
    print('enter same_len')
    print(args)
    g = groupby(args, lambda x: len(x))
    print(g)
    print('out')
    return next(g, True) and not next(g, False)

def in_tuple_to_numpy(fn: Callable):
    """decorator to convert input tuples to numpy"""
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

def flat_tuple(tup: Tuple[Union[Tuple[int, ...], int], ...]) -> Tuple[int, ...]:
    """flat a tuple made of int or tuples of int"""
    return tuple(chain(*(i if isinstance(i, tuple) else (i,) for i in tup)))

# utils
def format_padding(
        array_ndim: int,
        padding: Tuple[Union[Tuple[int, ...], int], ...]
    ) -> Tuple[Tuple[int, ...], ...]:
    """
    If an iterator is supplied to numpy.pad's pad_width it 
    must have the same length as the array that is going to 
    be padded and every element must be of the same type. 

    For a 2D array, pad_width = (1, (3, 2)) not accepted,
    pad_width = ((1,), (3, 2)) not accepted, pad_width =
    ((1,1), (3,1)) accepted. 
    """
    return tuple((0,0) for _ in range(array_ndim - len(padding))) \
        + tuple(
            i if isinstance(i, tuple) else (i, i)
            for i in padding
        )

@in_tuple_to_numpy
def output_shape(
        array_shape: Tuple[int],
        kernel_size: Tuple[int], # not (C_out, C_in, X0, X1, ...), just (X0, X1, ...)
        stride: Tuple[int], 
        dilation: Tuple[int],
        padding: Tuple[Tuple[int, ...], ...] # already formatted padding by format_padding
    ) -> Tuple[int]:
    """
    Compute output shape as a generalization of PyTorch output shape 
    formula for any dimension. For example, in 3d convolution, i.e.
    (N, C_in, D_in, H_in, W_in), D_out is as follows:

                  | D_in + 2*padding[0] - dilation[0] * (kernel_shape[0] - 1) - 1               |
    D_out = floor | ------------------------------------------------------------- (divide) + 1  |
                  |                             stride[0]                                       |
    
    If padding[i] is not symmetric, i.e. is a tuple of the amount of
    padding before and after that dimension, the sum of that tuple is 
    used instead.

    where 
        stride:   (strideD_in, strideH_in, strieW_in)
        dilation: (dilationD_in, ...H_in, ...W_in)
        padding:  (paddingD_in, ...H_in, ...W_in)
    """
    padding = padding.sum(axis=1)
    assert same_len(kernel_size, stride, dilation, padding), \
    "kernel_shape, stride, dilation and padding must have same length, got: \n" + \
    f"kernel_size: {kernel_size}\nstride: {stride}\ndilation: {dilation}\npadding: {padding}"
    print('kernel_size', kernel_size)
    print('stride', stride)
    print('dilation', dilation)
    print('padding', padding)
    conv_dims =len(kernel_size)
    ret = np.floor(
        (array_shape[-conv_dims:] + padding - dilation*(kernel_size - 1) - 1) / stride + 1
    )
    
    return tuple(ret.astype(int))

def kernel_ready_as_strided(
        array: NDArray, 
        kernel_size: Tuple[int], # not (C_out, C_in, X0, X1, ...), just (X0, X1, ...)
        stride: Tuple[int], 
        dilation: Tuple[int],
        output_shape: Tuple[int],
    ) -> NDArray:
    """
    Computes as_strided view of the input array in order to make 
    convolutions with tensordot easily. For example, a 2D convolution 
    with size (N, C_in, H_in, W_in) with some filter of size 
    (C_out, C_in, kH, kW) multiplies every fiter in C_out 
    H_out * W_out times. 

    The strided view will have dimensions (N, H_out, W_out, C_in, kH, kW),
    so every kernel will be multiplied element wise for every (..., H_out, W_out, ...)
    compatible view and reduced using sum (this is done by tensordot). 
    Note that the last three axes of (N, H_out, W_out, C_in, kH, kW) are compatible 
    with the three of (C_out, C_in, kH, kW), but only the last two are needed for 
    _kernel_ready_as_strided.

    How to calculate as_strided's shape and strides for 3D convolution,
    from innermost dimensions to outermost ones:

    - strides:

        [ -8, -7, -6 , -5 , -4 , -3 , -2 , -1 ] * itemsize in bytes
        
        position -1: 1 * dilation[2], horizontal dilation
        position -2: W_in * dilation[1], vertical dilation
        position -3: W_in * H_in * dilation[0], depth dilation
        position -4: W_in * H_in * D_in * 1, jump to next Channel
        position -5: 1 * stride[2], horizontal stride
        position -6: W_in * stride[1], vertical stride
        position -7: W_in * H_in * stride[0], depth stride
        position -8: W_in * H_in * D_in * C_in, next observation in batched data

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
    assert array.flags["C_CONTIGUOUS"], "_kernel_ready_as_strided only accepts contiguous arrays"

    # precompute some values
    C_in, noC_in_shape = array.shape[1], array.shape[2:]

    # compute decreasing factorial like pattern as in the example
    fact_pattern = [np.prod(noC_in_shape[i:], dtype=int) for i in range(len(noC_in_shape)+1)] * 2
    remaining_pattern = (C_in,) + stride + (1,) + dilation
    print('fact_pattern' ,fact_pattern)
    print('remaining_pattern', remaining_pattern)
    # calculate strides and the as_strided output shape, if online learning N value not needed
    strides = [a*b*array.itemsize for a, b in zip(fact_pattern, remaining_pattern)]
    shape = (array.shape[0],) + output_shape + (C_in, ) + kernel_size
    print('strides no byte: ', np.array(strides)/array.itemsize)
    print('strides in bytes: ', strides)
    print('strided output shape: ', shape)
    
    return as_strided(array, shape=shape, strides=strides)


@in_tuple_to_numpy
def trimm_uneven_stride(
        array: NDArray, # already padded
        kernel_size: Tuple[int], # not (C_out, C_in, X0, X1, ...), just (X0, X1, ...)
        stride: Tuple[int],
        dilation: Tuple[int]
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

        M = (kernel_size - 1) * (dilation - 1) + kernel_shape
    
    Therefore, if (sample_size - M) % stride != 0, filter does not
    cover the entire image and it must be sliced.
    """
    # precompute some values
    noC_in_shape = array.shape[2:]

    M = (kernel_size - 1) * (dilation - 1) + kernel_size 
    offset = (noC_in_shape - M) % stride 

    if np.any(offset): 
        slices = (..., ) + tuple(slice(i) for i in no_channels_shape-offset)
        return array[slices]

    return array



if __name__ == "__main__":
    import sys
    sys.path.append('../../')
    from giagrad.nn.layers.conv import ConvND
    from giagrad import Tensor
    import string
    from itertools import chain
    from math import floor
    np.random.seed(1234)
    alphabet = [chr(i) for i in chain(range(33,127), range(161, 1500))]
    # Channels out, Channels in, kernel Height, kernel Width
    def kernel(Cout, Cin, *args, a=0, b=4):
        return np.random.randint(a, b, (Cout, Cin)+(args))

    def dataBatched(*args):
        return np.array(alphabet[:np.prod(args)], dtype=object).reshape(args)
    #     return np.random.randint(-1000, 1000, (N, Cin, Hin, Win, Din))
    
    # define size
    N, Cin, Hin, Win = 1, 2, 3, 3
    kH, kW = (kernel_size := (5, 5))
    Cout = 2
    # DATA
    data = dataBatched(N, Cin, Hin, Win)
    # random KERNEL
    k = kernel(Cout, Cin, kH, kW)

    x = Tensor(data)
    w = Tensor(k)

    conv = ConvND(
        kernel_size=kernel_size,
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1,1),
        constant_values='0'
    )
    print('data\n', data)
    print('kernel\n', k)
    print('kernel shape', k.shape)
    conv.forward(x, w)

    #  [[[[3 3]
    #    [2 1]]

    #   [[0 0]
    #    [0 1]]]


    #  [[[3 1]
    #    [3 1]]

    #   [[2 2]
    #    [3 2]]]]
    # kernel shape (2, 2, 2, 2)
    # x [[[['0' '0' '0' '0' '0']
    #    ['0' '!' '"' '#' '0']
    #    ['0' '$' '%' '&' '0']
    #    ['0' "'" '(' ')' '0']
    #    ['0' '0' '0' '0' '0']]

    #   [['0' '0' '0' '0' '0']
    #    ['0' '*' '+' ',' '0']
    #    ['0' '-' '.' '/' '0']
    #    ['0' '0' '1' '2' '0']
    #    ['0' '0' '0' '0' '0']]]]

    # padding = (0, 0)
    # dilation = (2, 2)
    # stride = (1, 1)
    # trimmed_data = trimm_uneven_stride(
    #     data,
    #     kernel_size,
    #     stride,
    #     dilation
    # )
    # print('data shape', data.shape)
    # print(data)
    # print('trimmed data shape', trimmed_data.shape)
    # print(trimmed_data)

    # axis_pad = format_padding(
    #     data.ndim,
    #     padding
    # )
    # print(axis_pad)
    # out_shape = output_shape(
    #         data.shape,
    #         kernel_size,
    #         stride,
    #         dilation,
    #         axis_pad[2:]
    #     )

    # if not all(i > 0 for i in out_shape):
    #     msg = "Stride, dilation, padding and kernel dimensions are incompatible:\n"
    #     msg += f"Input dimensions: {data.shape}\n"
    #     msg += f"Kernel dimensions: {k.shape}\n"
    #     msg += f"Stride dimensions: {stride}\n"
    #     msg += f"Padding dimensions: {padding}\n"
    #     msg += f"Dilation dimensions: {dilation}\n"
    #     raise ValueError(msg) 

    # print('desired total out_shape', (N, Cout) + out_shape)

    # strided_array = kernel_ready_as_strided(
    #         data,
    #         kernel_size,
    #         stride,
    #         dilation,
    #         out_shape
    #     )

    # print('kernel shape: ', k.shape)
    # print('data shape: ', data.shape)

    # conv_dims = k.ndim - 1
    # ret = np.tensordot(
    #         k, 
    #         strided_array, 
    #         axes = [
    #             range(k.ndim)[-conv_dims:],
    #             range(strided_array.ndim)[-conv_dims:]
    #             ]
    #     )
    # ret = np.swapaxes(ret, 0, 1)
    # print(ret.shape)

