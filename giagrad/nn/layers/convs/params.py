from giagrad.tensor import Tensor
from giagrad.nn.layers.utils import flat_tuple, format_tuples, same_len
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
        'out_channels', 'stride', 'dilation', 'padding', 'padding_mode', 
        'padding_kwargs', 'groups', 'online_learning', 'axis_pad', 'kernel_size'
    )

    def __init__(
        self,
        kernel_size: Tuple[int, ...],
        stride: Union[Tuple[int, ...], int],
        dilation:  Union[Tuple[int, ...], int],
        padding: Union[Tuple[Union[Tuple[int, ...], int], ...], int],
        padding_mode: str,
        groups: int,
        padding_kwargs: Dict[str, Any],
    ):

        self.kernel_size = np.array(kernel_size)
        self.padding_mode = padding_mode
        self.padding_kwargs = padding_kwargs
        self.groups = groups
        self.online_learning = False
        self.axis_pad: Optional[NDArray] = None

        if isinstance(stride, int):
            stride = (stride,)*len(kernel_size)

        if isinstance(dilation, int):
            dilation = (dilation,)*len(kernel_size)

        if isinstance(padding, int):
            padding = (padding,)*len(kernel_size)

        if not (
            isinstance(stride, Iterable) 
            and isinstance(dilation, Iterable) 
            and isinstance(padding, Iterable)
        ):
            msg = "stride, dilation and padding must be an iterable or int, got:\n"
            msg += f"stride: {stride} "
            msg += f"dilation: {dilation} "
            msg += f"padding: {padding} "
            raise ValueError(msg)

        assert same_len(kernel_size, padding, dilation, stride),\
        f"kernel_size, padding, dilation and stride must have the same length"
        
        assert len(stride) == len(kernel_size) and all(
            s >= 1 and isinstance(s, int) for s in flat_tuple(stride)
        ), f"stride must have positive integers, got: {stride}"

        assert len(dilation) == len(kernel_size) and all(
            d >= 1 and isinstance(d, int) for d in flat_tuple(dilation) 
        ), f"dilation must have positive integers, got: {dilation}"

        assert len(padding) == len(kernel_size) and all(
            p >= 0 and isinstance(p, int) for p in flat_tuple(padding)
        ), f"padding must have non negative integers, got: {padding}"

        self.stride = np.array(stride)
        self.dilation = np.array(dilation)
        self.padding = np.array(format_tuples(padding))

    @property
    def needs_padding(self) -> bool:
        return np.sum(self.padding).item() != 0

    @property
    def conv_dims(self) -> int:
    	return len(self.kernel_size)

    def set_axis_pad(self, x: Tensor):
    	self.axis_pad = np.append(
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