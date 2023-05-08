from giagrad.tensor import Tensor, Function
from giagrad.nn.layers.convs.params import ConvParams 
from giagrad.nn.layers.convs.utils import output_shape, trimm_extra_padding
from giagrad.nn.layers.convs.ops import convolve, transpose
from giagrad.nn import Module

import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, Optional, Dict, Any, List

class _ConvND(Function):
    def __init__(self, params: ConvParams):
        super().__init__()
        self.params = params

    def forward(self, x: Tensor, w: Tensor):
        self.save_for_backward(x, w)
        return convolve(x, w, self.params)

    def backward(self, partial: NDArray):
        (xt, wt), partialt = self.parents, Tensor(partial)

        # DERIVATIVE w.r.t x, i.e. input tensor
        # partial (convolve) filters/kernels
        if xt.requires_grad:
            xt.grad += transpose(
                Tensor(trimm_extra_padding(partialt.data, self.params)),
                wt, 
                self.params)
        
        # DERIVATIVE w.r.t w, i.e. kernels/filters
        # x (convolve) partial
        if wt.requires_grad:
            params = self.params.copy()
            params.kernel_size = np.array(partial.shape)[-params.conv_dims:]
            params.swap_stride_dilation()

            # w_partial has shape (C_out, X0_out, x1_out, ..., C_in) or
            # (N, C_out, X0_out, x1_out, ..., C_in) 
            w_partial = convolve(xt, partialt, params, backward=self.params)
            if not params.online_learning:
                # sum across N batches
                w_partial = np.sum(w_partial, axis=0) # do not keep dims

            # move C_in to 1st position
            w_partial = np.rollaxis(w_partial, -1, 1) 
            wt.grad += w_partial


class ConvND(Module):

    __valid_input_dims: List[int]
    __error_message: str

    def __init__(
        self, 
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Union[Tuple[int, ...], int] = 1, 
        dilation:  Union[Tuple[int, ...], int] = 1, 
        padding: Union[Tuple[Union[Tuple[int, int], int], ...], int] = 0,
        padding_mode: str = 'constant',
        groups: int = 1, # TODO, add functionality
        bias: bool = True,
        **padding_kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.bias = bias

        self.__params = ConvParams(
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode=padding_mode,
            groups=groups,
            padding_kwargs=padding_kwargs
        )

    def __init_tensors(self, x: Tensor):
        # init tensors, only C_in is needed
        if x.ndim not in self.__valid_input_dims:
            raise ValueError(self.__error_message.format(shap=x.shape))

        in_channels = x.shape[0] if x.ndim == 3 else x.shape[1]
        k = np.sqrt(self.groups / (in_channels * np.prod(self.kernel_size)))

        self.w = Tensor.empty(
            self.out_channels, 
            in_channels, 
            *self.kernel_size, 
            requires_grad=True
        ).uniform(-k, k)
        
        if self.bias:
            self.b = Tensor.empty(
                *output_shape(x.shape, self.__params), 
                requires_grad=True 
            ).uniform(-k, k)

    def __call__(self, x: Tensor) -> Tensor:
        # initialize weights and bias if needed
        self.__initialize_tensors(x)
        # ConvND needs to know if online learning is done
        if x.ndim == self.__valid_input_dims[0]:
            self.__params.online_learning = True
        # update padding so that numpy.pad accepts it
        self.__params.set_axis_pad(x)
        # output tensor
        conv = Tensor.comm(_ConvND(self.__params), x, self.w)
        if self.bias:
            return conv + self.b 
        return conv

    def __str__(self):
        # TODO
        return f"{type(self)}()"