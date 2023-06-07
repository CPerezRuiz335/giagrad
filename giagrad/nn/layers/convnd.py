from giagrad.tensor import Tensor, Function
from giagrad.nn.layers.utils import (
    conv_output_shape, 
    trimm_uneven_stride,
    check_parameters, 
    sliding_filter_view,
    transpose,
    padding_same
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
        self._name = f"Conv{len(stride)}D"

    def forward(self, x: Tensor, w: Tensor):
        self.save_for_backward(x, w)
        conv_dims = w.ndim-2
        # make a view of x ready for tensordot
        sliding_view = sliding_filter_view(
            array=x.data,
            kernel_size=w.shape[-conv_dims:],
            stride=self.stride,
            dilation=self.dilation,
        )
        # convolve the last conv_dims dimensions of w and sliding_view    
        axes = [[-(axis+1) for axis in range(conv_dims+1)],]*2
        conv_out = np.tensordot(w.data, sliding_view, axes=axes)
        # (C_out, N, W0, ...) -> (N, C_out, W0, ...)
        conv_out = np.swapaxes(conv_out, 0, 1)

        if not conv_out.flags['C_CONTIGUOUS']:
            return np.ascontiguousarray(conv_out)
        return conv_out

    def backward(self, partial: NDArray):
        x, w = self.parents
        trimm_kwargs = {
            'kernel_size': w.shape[2:],
            'stride': self.stride,
            'dilation': self.dilation
        }

        # differentiate w.r.t inputer tensor
        if x.requires_grad:
            trimm_grad = trimm_uneven_stride(x.grad, **trimm_kwargs)
            trimm_grad += transpose(
                x=partial,
                w=w.data, 
                stride=self.stride,
                dilation=self.dilation
            )
        
        # differentiate w.r.t weights
        if w.requires_grad:
            conv_dims = partial.ndim-2
            trimm_data = trimm_uneven_stride(x.data, **trimm_kwargs)
            sliding_view = sliding_filter_view(
                array=trimm_data,
                kernel_size=partial.shape[-conv_dims:],
                stride=self.dilation,
                dilation=self.stride,
            )
            # convolve the last conv_dims dimensions of w, 
            # sliding_view, and batch dimension    
            axes = [[0] + [-(axis+1) for axis in range(conv_dims)],]*2
            w_partial = np.tensordot(partial, sliding_view, axes=axes)
            # w_partial has shape (N, C_out, X0_out, x1_out, ..., C_in) 
            w.grad += np.rollaxis(w_partial, -1, 1)  # move C_in to 1st position

extend = lambda x, len_: (x,)*len_ if isinstance(x, int) else x

class ConvND(Module):

    _valid_input_dims: List[int]
    _error_message: str

    def __init__(
        self, 
        out_channels: int,
        kernel_size: Union[Tuple[int, ...], int],
        stride: Union[Tuple[int, ...], int] = 1, 
        dilation:  Union[Tuple[int, ...], int] = 1, 
        padding: Union[Union[Tuple[Union[Tuple[int, int], int], ...], int], str] = 0,
        padding_mode: str = 'constant',
        groups: int = 1, # TODO, add functionality
        bias: bool = True,
        **padding_kwargs
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        
        check_parameters(
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding
        )

        conv_dims = len(kernel_size)
        
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = extend(stride, conv_dims)
        self.dilation = extend(dilation, conv_dims)
        self.padding = extend(padding, conv_dims)
        self.groups = groups
        self.padding_mode = padding_mode
        self.padding_kwargs = padding_kwargs
        self.bias = bias    

    def __init_tensors(self, x: Tensor, output_shape: NDArray):
        """init tensors, only C_in is needed"""
        if x.ndim not in self._valid_input_dims:
            raise ValueError(self._error_message.format(shape=x.shape))

        in_channels = x.shape[0] if x.ndim == self._valid_input_dims[0] else x.shape[1]
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
        if self.padding == 'same':
            padding = padding_same(
                array_shape=x.shape,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation
            )
        else:
            padding = self.padding

        output_shape = conv_output_shape(
            array_shape=x.shape, 
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=padding
        )  

        if not np.all(output_shape > 0):
            raise ValueError(
                "Stride, dilation, padding and kernel dimensions are incompatible"
            )  

        self.__init_tensors(x, output_shape)
        online_learning = x.ndim == self._valid_input_dims[0]

        x = x.unsqueeze(axis=0) if online_learning else x
        x = x.pad(*padding, mode=self.padding_mode, **self.padding_kwargs)
        x = Tensor.comm(_ConvND(self.stride, self.dilation), x, self.w)
        x = x.squeeze(axis=0) if online_learning else x 
        return x + self.b if self.bias else x

    def __str__(self):
        return (
            f"{type(self).__name__}("
            + f"{self.out_channels}, "
            + f"kernel_size={self.kernel_size}, "
            + f"stride={self.stride}, "
            + f"dilation={self.dilation}, "
            + f"padding={self.padding}, "
            + f"bias={self.bias}"
            + (f", groups={self.groups}" if self.groups > 1 else '')
            + ')'
        )

class Conv1D(ConvND):
    r"""1D covolution layer.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})` can be
    precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {{'same'}}, a tuple of ints or tuples of two ints
      giving the amount of implicit padding applied on both sides.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    {groups_note}

    Note:
        {depthwise_separable_note}

    Note:
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.

    Note:
        This module supports complex data types i.e. ``complex32, complex64, complex128``.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to both sides of
            the input. Default: 0
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})` or :math:`(C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels},
            \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``, then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`

    Examples::

        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    _valid_input_dims = [2, 3]
    _error_message  = "Conv1D only accepts input tensors of shapes:\n"
    _error_message += "\t(N, C_in, W_in) or (C_in, W_in)\n"
    _error_message += "\tgot: {shape}"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Conv2D(ConvND):
    _valid_input_dims = [3, 4]
    _error_message  = "Conv2D only accepts input tensors of shapes:\n"
    _error_message += "\t(N, C_in, H_in, W_in) or (C_in, H_in, W_in)\n"
    _error_message += "\tgot: {shape}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Conv3D(ConvND):
    _valid_input_dims = [4, 5]
    _error_message  = "Conv3D only accepts input tensors of shapes:\n"
    _error_message += "\t(N, C_in, D_in, H_in, W_in) or (C_in, D_in, H_in, W_in)\n"
    _error_message += "\tgot: {shape}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
