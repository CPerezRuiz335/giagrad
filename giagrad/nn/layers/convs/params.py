from giagrad.tensor import Tensor
from giagrad.nn.layers.convs.utils import flat_tuple, tuple_pairs, same_len
import numpy as np
from numpy.typing import NDArray
import copy
from collections.abc import Iterable
from typing import Tuple, Union, Dict, Optional, Any, Callable, Optional

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
    __slots__ = (
        'out_channels', 'stride', 'dilation', 'padding', 'padding_mode', 'conv_dims',
        'padding_kwargs', 'groups', 'online_learning', 'axis_pad', 'kernel_size',
        'needs_padding'
    )

    def __init__(
        self,
        kernel_size: Tuple[int, ...],
        stride: Union[Tuple[int, ...], int],
        dilation:  Union[Tuple[int, ...], int],
        padding,
        groups: int,
        padding_kwargs: Dict[str, Any] = dict(),
        padding_mode: str = 'constant'
    ):
        self.kernel_size = np.array(kernel_size)
        self.padding_mode = padding_mode
        self.padding_kwargs = padding_kwargs
        self.groups = groups
        self.axis_pad: Optional[NDArray] = None

        if isinstance(stride, int):
            stride_ = (stride,)*len(kernel_size)

        if isinstance(dilation, int):
            dilation_ = (dilation,)*len(kernel_size)

        if isinstance(padding, int):
            padding_ = (padding,)*len(kernel_size)

        if not (
            isinstance(stride_, Iterable) 
            and isinstance(dilation_, Iterable) 
            and isinstance(padding_, Iterable)
        ):
            msg = "stride, dilation and padding must be an iterable or int, got:\n"
            msg += f"stride: {stride} "
            msg += f"dilation: {dilation} "
            msg += f"padding: {padding} "
            raise ValueError(msg)

        assert same_len(kernel_size, padding_, dilation_, stride_),\
        f"kernel_size, padding, dilation and stride must have the same length"
        
        assert len(stride_) == len(kernel_size) and all(
            s >= 1 and isinstance(s, int) for s in flat_tuple(stride_)
        ), f"stride must have positive integers, got: {stride}"

        assert len(dilation_) == len(kernel_size) and all(
            d >= 1 and isinstance(d, int) for d in flat_tuple(dilation_) 
        ), f"dilation must have positive integers, got: {dilation}"

        assert len(padding_) == len(kernel_size) and all(
            p >= 0 and isinstance(p, int) for p in flat_tuple(padding_)
        ), f"padding must have non negative integers, got: {padding}"

        self.stride = np.array(stride_)
        self.dilation = np.array(dilation_)
        self.padding = np.array(tuple_pairs(padding_))
        self.conv_dims = len(kernel_size)
        self.needs_padding = np.sum(self.padding).item() != 0

    def axis_pad(self, x: NDArray):
        return np.append(
            ((0,0),)*(x.ndim - self.conv_dims), 
            self.padding, 
            axis=1
        )

    def swap_stride_dilation(self):
        self.stride, self.dilation = self.dilation, self.stride
        return self

    def copy(self):
        # shallow copy is enough
        return copy.copy(self)