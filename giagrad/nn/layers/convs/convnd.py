from giagrad.tensor import Tensor, Function
from giagrad.nn.layers.convs.utils import (
    conv_output_shape, 
    trimm_uneven_stride,
    check_parameters, 
    format_padding, 
    extend,
    convolve,
    transpose
)
from giagrad.nn import Module
import numpy as np
from numpy.typing import NDArray
from typing import Union, Tuple, Optional, Dict, Any, List

class _ConvND(Function):
    def __init__(self, stride: Tuple[int, ...], dilation: Tuple[int, ...]):
        super().__init__()
        self.stride = stride 
        self.dilation = dilation

    def forward(self, x: Tensor, w: Tensor):
        self.save_for_backward(x, w)
        out = convolve(x.data, w.data, self.stride, self.dilation)
        if not out.flags['C_CONTIGUOUS']:
            return np.ascontiguousarray(out)
        return out

    def backward(self, partial: NDArray):
        x, w = self.parents

        # differentiate w.r.t inputer tensor
        if x.requires_grad:
            trimm_kwargs = {
                'kernel_size': w.shape[2:],
                'stride': self.stride,
                'dilation': self.dilation
            }
            trimm_grad = trimm_uneven_stride(x.grad, **trimm_kwargs)
            trimm_grad += transpose(
                x=partial,
                w=w.data, 
                stride=self.stride,
                dilation=self.dilation
            )
        
        # differentiate w.r.t weights
        if w.requires_grad:
            trimm_data = trimm_uneven_stride(
                array=x.data,
                kernel_size=w.shape[2:],
                stride=self.stride,
                dilation=self.dilation
            )

            w_partial = convolve(
                x=trimm_data, 
                w=partial, 
                stride=self.dilation, 
                dilation=self.stride
            )
            # w_partial has shape (N, C_out, X0_out, x1_out, ..., C_in) 
            w.grad += np.rollaxis(w_partial, -1, 1)  # move C_in to 1st position


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
        check_parameters(
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding
        )

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.padding_kwargs = padding_kwargs
        self.bias = bias    

    def __init_tensors(self, x: Tensor, output_shape: NDArray):
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
                *output_shape, 
                requires_grad=True 
            ).uniform(-k, k)

    def __call__(self, x: Tensor) -> Tensor:
        conv_dims = len(self.kernel_size)
        output_shape = conv_output_shape(
            array_shape=x.shape, 
            kernel_size=self.kernel_size,
            stride=extend(self.stride, conv_dims),
            dilation=extend(self.dilation, conv_dims),
            padding=format_padding(self.padding, conv_dims=conv_dims)
        )  

        if not np.all(output_shape > 0):
            raise ValueError(
                "Stride, dilation, padding and kernel dimensions are incompatible"
            )  

        self.__initialize_tensors(x, output_shape)
        online_learning = x.ndim == self.__valid_input_dims[0]

        x = x.unsqueeze(axis=0) if online_learning else x
        x = x.pad(self.padding, self.padding_mode, **self.padding_kwargs)
        x = Tensor.comm(
            _ConvND(self.__parameters.stride, self.__parameters.dilation), x, self.w
        )
        x = x.squeeze(axis=0) if online_learning else x 
        return x + self.b if self.bias else x

    def __str__(self):
        # TODO
        return f"{type(self)}()"