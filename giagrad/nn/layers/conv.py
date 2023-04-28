import numpy as np
import giagrad.nn.layers.utils as utils 
from giagrad.tensor import Tensor, Function
from typing import Union, Tuple
from numpy.typing import NDArray

# def __initalize_tensors(self, in_channels: int):
#   stdev = np.sqrt(self.groups / (in_channels * np.prod(self.kernel_size)))
#   self.w = 

def ConvND(Function):
    def __init__(
        self, 
        out_channels: int, 
        kernel_size: Union[Tuple[int, ...], int],
        stride: Union[Tuple[int, ...], int],
        padding: Union[Tuple[int, ...], int],
        dilation: Union[Tuple[int, ...], int],
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'constant',
        online_learning: bool = False
    ):
        # variables
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.bias = bias
        self.groups = groups
        self.stride = stride if isinstance(stride, tuple) else (stride,)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,)
        self.padding_mode = padding_mode
        if isinstance(padding, tuple):
            self.padding = tuple(
                i if isinstance(i, tuple) else (i,) 
                for i in padding
            )
        else:
            self.padding = (padding, )
        self.online_learning = online_learning

        # check values
        assert len(self.stride) == len(self.kernel_size) and all(
            s >= 1 and isinstance(s, int) for s in self.stride 
        ), f"stride must have positive integers, got: {stride}"

        assert len(self.dilation) == len(self.kernel_size) and all(
            d >= 1 and isinstance(d, int) for d in self.dilation 
        ), f"dilation must have positive integers, got: {dilation}"

        assert len(self.padding) == len(self.kernel_size) and all(
            p >= 0 and isinstance(p, int) for p in sum(self.padding, ())
        ), f"padding must non negative integers, got: {padding}"


    def forward(self, x, w):
        self.save_for_backward(x, w)
        # x ... data:       (N, C, X0, X1, ...)
        # filters ... data: (F, C, W0, W1, ...)

        x = x.data
        w = w.data
        conv_dims = w.ndim - 2

        # expected output shape 
        output_shape = utils._output_shape(
            x.shape,
            w.shape,
            self.stride,
            self.dilation,
            self.padding[-conv_dims:]
        )     

        if not all(i > 0 for i in output_shape):
            msg = "Stride, dilation, padding and kernel dimensions are incompatible:\n"
            msg += f"Input dimensions: {x.shape}\n"
            msg += f"Kernel dimensions: {w.shape}\n"
            msg += f"Stride dimensions: {self.stride}\n"
            msg += f"Padding dimensions: {self.padding[-conv_dims:]}\n"
            msg += f"Dilation dimensions: {self.dilation}\n"
            raise ValueError(msg)   

        # extend padding to all axis
        if any(sum(self.padding, ())):
            axis_pad = (0,) * (2-self.online_learning) + self.padding
            x = np.pad(x, axis_pad, mode=self.padding_mode) 

        # make a view of x ready for tensordot
        # see ./util.py
        kernel_ready_view = utils._kernel_ready_as_strided(
            x,
            w.shape,
            self.stride,
            self.dilation,
            output_shape,
            self.online_learning
        )

        # main convolution
        conv_out = np.tensordot(
            w, 
            kernel_ready_view, 
            axes = [
                range(w.ndim)[-conv_dims:],
                range(kernel_ready_view.ndim)[-conv_dims:]
            ]
        )

        if self.online_learning:
            return conv_out
        return np.swapaxes(conv_out, 0, 1)
