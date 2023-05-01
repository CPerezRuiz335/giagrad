import numpy as np
import giagrad.nn.layers.conv_utils as utils 
from giagrad.nn.layers.conv_utils import ConvParams 
from giagrad.tensor import Tensor, Function
from typing import Union, Tuple, Optional, Dict, Any
from numpy.typing import NDArray
from giagrad.nn import Module
from dataclasses import dataclass, fields, asdict


class _ConvND(Function):
    def __init__(self, parameters: ConvParams):
        super().__init__()
        self.params = parameters

    # single convolution
    @staticmethod
    def convolve(
            xt: Tensor, wt: Tensor, params: ConvParams, forward: bool = True
        ) -> NDArray:
        """
        Convolves xt (*) wt, if forward equals False wt is the output tensor's
        derivative os shape is not (C_out, C_in, W0, W1, ...), is just
        (C_out, W0, W1, ...) or (N, C_out, W0, W1, ...).
        """
        x, w = xt.data, wt.data
        # if online learning add N=1, where N: (N, ...)
        if params.online_learning:
            x = np.expand_dims(x, axis=0)
            if not forward:
                w = np.expand_dims(w, axis=0)

        # output tensor shape 
        output_shape = params.output_shape(x.shape)     

        if not all(i > 0 for i in output_shape):
            raise ValueError(
                "Stride, dilation, padding and kernel dimensions are incompatible"
                )   

        # apply padding if at least one non-zero entry
        if np.sum(params.padding).item():
            x = np.pad(
                x, params.axis_pad, 
                mode=params.padding_mode, 
                **params.padding_kwargs
            ) 
        
        # make a view of x ready for tensordot
        kernel_ready_view = utils.kernel_ready_view(array=x, params=params)

        if forward:
            axes = [[-i for i in reversed(range(1, params.conv_dims+1))]]*2
        else:
            axes = [[0]+list(-i for i in reversed(range(1, params.conv_dims+1)))]

        conv_out = np.tensordot(w, kernel_ready_view, axes=axes) # type: ignore
        # (C_out, N, W0, ...) -> (N, C_out, W0, ...)
        conv_out = np.swapaxes(conv_out, 0, 1)
        
        if not conv_out.flags['C_CONTIGUOUS']:
            conv_out = np.ascontiguousarray(conv_out)

        if params.online_learning:
            return np.squeeze(conv_out, axis=0) # inverse of expand_dims
        return conv_out

    def forward(self, x: Tensor, w: Tensor):
        self.save_for_backward(x, w)
        return _ConvND.convolve(x, w, self.params)

    def backward(self, partial: NDArray):
        (xt, wt), partialt = self.parents, Tensor(partial)
        # if filter does not cover entire sample
        # trimm_uneven_stride will pad and reshape 
        x_trimm = utils.trimm_uneven_stride(
            array=xt.data,
            params=self.params
        )
        xt_trimm = Tensor(x_trimm)

        # DERIVATIVE w.r.t w, i.e. kernels/filters
        # x (convolve) partial
        params = self.params.copy()
        params.kernel_size = np.array(partial.shape)[-params.conv_dims:]
        params.dilation, params.stride = params.stride, params.dilation
        # trimm_uneven_stride already padded x
        params.axis_pad = np.zeros_like(params.axis_pad)
        params.padding = np.zeros_like(params.padding)
        w_partial = _ConvND.convolve(xt_trimm, partialt, params, forward=False)
        # w_partial has shape (C_out, X0_out, x1_out, ..., C_in)
        w_partial = np.rollaxis(w_partial, -1, 1) # move C_in to 1st position
        wt.grad += w_partial

        # DERIVATIVE w.r.t x, i.e. input tensor
        # x (convolve) partial


class Conv2D(Module):
    __slots = [
        'out_channels', 'kernel_size', 'stride',
        'dilation', 'padding', 'groups', 'bias'
    ]

    def __init__(
        self, 
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Union[Tuple[int, ...], int], 
        dilation:  Union[Tuple[int, ...], int], 
        padding: Union[Tuple[Union[Tuple[int, int], int], ...], int],
        padding_mode: str = 'constant',
        groups: int = 1, # TODO, add functionality
        bias: bool = True,
        **padding_kwargs
    ):
        super().__init__()
        local_dict = locals().copy()
        local_dict.pop('self')
        local_dict.pop('__class__')
        self.params = ConvParams(**local_dict)

    def __getattr__(self, attr: str):
        if attr not in self.__slots:
            raise AttributeError(f"{attr}")
        try:
            out_attr = getattr(self.params, attr)
        except AttributeError:
            raise AttributeError(f"{attr}")
        return out_attr

    def __initialize_tensors(self, x: Tensor):
        # init tensors, only C_in is needed
        if x.ndim not in [3, 4]:
            msg = "Conv2D only accepts input tensors of shapes:\n"
            msg += "(N, C_in, H_in, W_in) or (C_in, H_in, W_in)\n"
            msg += f"got: {x.shape}"
            raise ValueError(msg)

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
                *self.params.output_shape(x.shape), 
                requires_grad=True 
            ).uniform(-k, k)

    def __call__(self, x: Tensor) -> Tensor:
        # initialize weights and bias if needed
        self.__initialize_tensors(x)
        # ConvND needs to know if online learning is done
        if x.ndim == 3:
            self.params.online_learning = True
        # update padding so that numpy.pad accepts it
        self.params.set_axis_pad(x)
        # output tensor
        conv = Tensor.comm(_ConvND(self.params), x, self.w)
        if self.bias:
            return conv + self.b 
        return conv

    def __str__(self):
        # TODO
        return f"Conv2D(, , ,)"



