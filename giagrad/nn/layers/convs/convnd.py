from giagrad.tensor import Tensor, Function
from giagrad.nn.layers.convs.params import ConvParams 
from giagrad.nn.layers.convs.utils import output_shape, trimm_extra_padding
from giagrad.nn.layers.convs.ops import convolve, transpose
from giagrad.nn import Module

import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, Optional, Dict, Any, List

class _ConvND(Function):
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.params = params

    def forward(self, x: Tensor, w: Tensor):
        self.save_for_backward(x, w)
        return convolve(x.data, w.data, self.params)

    def backward(self, partial: NDArray):
        xt, wt = self.parents

        # DERIVATIVE w.r.t x, i.e. input tensor
        # partial (convolve) filters/kernels
        if xt.requires_grad:
            xt.grad += transpose(
                trimm_extra_padding(partial, self.params),
                wt.data, 
                self.params)
        
        # DERIVATIVE w.r.t w, i.e. kernels/filters
        # x (convolve) partial
        if wt.requires_grad:
            params = self.params.copy()
            params.kernel_size = np.array(partial.shape)[-params.conv_dims:]
            params.swap_stride_dilation()

            # w_partial has shape (C_out, X0_out, x1_out, ..., C_in) or
            # (N, C_out, X0_out, x1_out, ..., C_in) 
            w_partial = convolve(xt.data, partial, params, backward=self.params)
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
        # save for ConvND    
        self.__parameters = locals().copy()
        del self.__parameters['self'] 
        utils.check_errors(**self.__parameters) # TODO from utils, replace class ConvParams

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.bias = bias    


    def __init_tensors(self, x: Tensor):
        """init tensors, only C_in is needed"""
        if x.ndim not in self.__valid_input_dims:
            raise ValueError(self.__error_message.format(x.shape))

        in_channels = x.shape[0] if x.ndim == self.__valid_input_dims[0] else x.shape[1]
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
        self.__initialize_tensors(x)
        online_learning = x.ndim == self.__valid_input_dims[0]

        x = x.unsqueeze(axis=0) if online_learning else x
        x = Tensor.comm(_ConvND(self.__parameters), x, self.w)
        x = x.squeeze(axis=0) if online_learning else x 
        return x + self.b if self.bias else x
        # if np.sum(self.padding):
        #     x = x.pad(*self.__params.padding, mode=self.mode, **self.padding_kwargs)
        # needs_trimm, idx = check_uneven_strides(self.__params)
        # if needs_trimm:
        #     x = x[idx]

    def __str__(self):
        # TODO
        return f"{type(self)}()"