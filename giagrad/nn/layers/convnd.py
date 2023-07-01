from typing import Tuple, Dict

import numpy as np
from numpy.typing import NDArray

from giagrad.nn import Module
from giagrad.tensor import Tensor, Function
from giagrad.nn.layers.utils import (
    conv_output_shape, 
    trimm_uneven_stride,
    check_parameters, 
    sliding_filter_view,
    convolve_forward,
    convolve_backward,
    transpose,
    padding_same
)

class _ConvND(Function):
    def __init__(
        self, 
        stride: Tuple[int, ...], 
        dilation: Tuple[int, ...],
        groups: int
    ):
        super().__init__()
        self.stride = stride 
        self.dilation = dilation
        self.groups = groups
        self._name = f"Conv{len(stride)}D"

    def forward(self, x: Tensor, w: Tensor):
        self.save_for_backward(x, w)

        if self.groups > 1:
            split_x = np.split(x.data, self.groups, axis=1)
            split_w = np.split(w.data, self.groups, axis=0)
            convolved_groups = [
                convolve_forward(sx, sw, self.stride, self.dilation)
                for sx, sw in zip(split_x, split_w)
            ]
            # save for backwards
            self.__split_w = split_w
            return np.concatenate(convolved_groups, axis=1)

        else:
            return convolve_forward(x.data, w.data, self.stride, self.dilation)

    def backward(self, partial: NDArray):
        x, w = self.parents 
        stride, dilation, groups = self.stride, self.dilation, self.groups

        trimm_kwargs = {
            'kernel_size': w.shape[2:],
            'stride': stride,
            'dilation': dilation
        }

        # split partial for x and w, at least w will need it
        if groups > 1: 
            split_partial = np.split(partial, groups, axis=1)

        # differentiate w.r.t input tensor
        if x.requires_grad and groups > 1:
            transposed_groups = []
            for sp, sw in zip(split_partial, self.__split_w):
                tmp = transpose(sp, sw, stride, dilation)
                transposed_groups.append(tmp)

            trimm_grad = trimm_uneven_stride(x.grad, **trimm_kwargs)
            trimm_grad += np.concatenate(transposed_groups, axis=1)
        
        if x.requires_grad and groups == 1:
            trimm_grad = trimm_uneven_stride(x.grad, **trimm_kwargs)
            trimm_grad += transpose(partial, w.data, stride, dilation)
        
        # differentiate w.r.t weights
        if w.requires_grad and groups > 1:
            trimm_x = trimm_uneven_stride(x.data, **trimm_kwargs)
            split_x = np.split(trimm_x, groups, axis=1)

            convolved_groups = []
            for sx, sp in zip(split_x, split_partial):
                tmp = convolve_backward(sx, sp, stride, dilation)
                convolved_groups.append(tmp)

            w.grad += np.concatenate(convolved_groups, axis=0)

        if w.requires_grad and groups == 1:
            trimm_x = trimm_uneven_stride(x.data, **trimm_kwargs)
            w.grad += convolve_backward(trimm_x, partial, stride, dilation)

extend = lambda x, len_: (x,)*len_ if isinstance(x, int) else x

class ConvND(Module):

    _valid_input_dims: Tuple[int, ...]
    _error_message: str
    _conv_dims: int

    def __init__(
        self, 
        out_channels: int,
        kernel_size: Tuple[int, ...] | int,
        stride: Tuple[int, ...] | int = 1, 
        dilation:  Tuple[int, ...] | int = 1, 
        padding: Tuple[Tuple[int, int] | int, ...] | int | str = 0,
        padding_mode: str = 'constant',
        groups: int = 1,
        bias: bool = True,
        **padding_kwargs
    ):
        super().__init__()
        self.kernel_size = extend(kernel_size, self._conv_dims)
        check_parameters(
            self.kernel_size, stride, dilation, padding, groups, out_channels
        )

        self.out_channels = out_channels 
        self.stride = extend(stride, self._conv_dims)
        self.dilation = extend(dilation, self._conv_dims)
        self.padding = extend(padding, self._conv_dims)
        if self._conv_dims == 1 and isinstance(padding, tuple) and len(padding) == 2:
            self.padding = (self.padding, )
        self.groups = groups
        self.padding_mode = padding_mode
        self.padding_kwargs = padding_kwargs
        self.bias = bias    

    def __init_tensors(self, x: Tensor, output_shape: NDArray):
        """init tensors, only C_in is needed"""
        if x.ndim not in self._valid_input_dims:
            raise ValueError(self._error_message.format(shape=x.shape))

        in_channels = x.shape[x.ndim == self._valid_input_dims[1]]

        if in_channels % self.groups != 0:
            raise ValueError(
                f"input channels={in_channels} are not " 
                f"divisible by groups={self.groups}"
            )

        k = np.sqrt(self.groups / (in_channels * np.prod(self.kernel_size)))

        self.w = Tensor.empty(
            self.out_channels, in_channels//self.groups, *self.kernel_size, 
            requires_grad=True
        ).uniform(-k, k)
        
        self.b = Tensor.empty(
            x.shape[0], self.out_channels, *output_shape, 
            requires_grad=True
        ).uniform(-k, k) if self.bias else None

    def __call__(self, x: Tensor) -> Tensor:
        padding = padding_same(
            x.shape, self.kernel_size, self.stride, self.dilation
        ) if self.padding == 'same' else self.padding

        output_shape = conv_output_shape(
            x.shape, self.kernel_size, self.stride, self.dilation, padding
        )  

        if not np.all(output_shape > 0):
            raise ValueError(
                "Stride, dilation, padding and kernel dimensions are incompatible"
            )  

        self.__init_tensors(x, output_shape)
        online_learning = x.ndim == self._valid_input_dims[0]

        x = x.unsqueeze(axis=0) if online_learning else x
        x = x.pad(*padding, mode=self.padding_mode, **self.padding_kwargs)
        f = _ConvND(self.stride, self.dilation, self.groups) 
        x = Tensor.comm(f, x, self.w)
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
    r"""
    1D convolution layer.
    
    Adapted from `PyTorch Conv1d`_. Note that this implementation 
    matches `PyTorch LazyConv1d`_, so **in_channels** is inferred
    at runtime.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, L)` and output 
    :math:`(N, C_{\text{out}}, L_{\text{out}})` can be
    precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.

    * :attr:`stride` 
        Controls the stride for the cross-correlation, a single number 
        or a one-element tuple.

    * :attr:`padding` 
        Controls the amount of padding applied to the input. It can be 
        either a string {'same'}, a tuple of ints or tuples of two 
        ints (at the same time) giving the amount of implicit padding 
        applied on both sides. Internally calls :meth:`giagrad.Tensor.pad`.

    * :attr:`dilation` 
        Controls the spacing between the kernel points; also known as 
        the à trous algorithm. It is harder to describe, but this 
        `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` 
        Controls the connections between inputs and outputs. 
        ``in_channels`` and :attr:`out_channels` must both be divisible 
        by :attr:`groups`. For example,
        
        - At groups = 1
            all inputs are convolved to all outputs.

        - At groups = 2 
            the operation becomes equivalent to having two 
            conv layers side by side, each seeing half the input channels
            and producing half the output channels, and both subsequently
            concatenated.

        - At groups = ``in_channels`` 
            each input channel is convolved 
            with its own set of filters (of size 
            :math:`\frac{\text{out_channels}}{\text{in_channels}}`).
      
    See `Animated AI`_ for an outstanding explanation.

    Note
    ----
    When `groups == in_channels` and  `out_channels == K * in_channels`, 
    where `K` is a positive integer, this operation is also known as a 
    "depthwise convolution".

    In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
    a depthwise convolution with a depthwise multiplier `K` can be 
    performed with the arguments 
    :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`.
    See `Animated AI`_.

    Note
    ----
    ``padding='valid'`` is the same as padding=0, which is the 
    default value. However ``padding='same'`` pads the input so the 
    output has the shape as the input. However, in some cases 
    padding before and after may vary, mainly due tue assymetrical 
    kernel sizes. For example, the formula to compute padding for a 
    given dimension :math:`X_{in}`:

    .. math::
        \begin{align*}
            padding &= \frac{
                (X_{in}-1) \times \text{stride} - X_{in} + K_{X_{in}} 
                + (K_{X_{in}}-1)*(\text{dilation}-1)
            }{2} \\
            pad_{before} &= \lceil padding \rceil  \\
            pad_{after}  &= \lfloor padding \rfloor  \\
        \end{align*}
                                               
    Parameters
    ----------
    out_channels: int
        Number of channels produced by the convolution.
    kernel_size: int or Tuple[int]
        Size of the convolving kernel.
    stride: int or Tuple[int], default: 1 
        Stride of the convolution. 
    dilation: int or Tuple[int], default: 1 
        Spacing between kernel elements. 
    padding: int, tuple or str, default: 0 
        Padding added to both sides of the input. See 
        :meth:`giagrad.Tensor.pad`.
    padding_mode: str, default: 'constant'
        Padding mode defined by `numpy.pad`_.
    groups: int, default: 1 
        Number of blocked connections from input channels to output 
        channels.
    bias: bool, default: ``True``
        If ``True``, adds a learnable bias to the output.
    **padding_kwargs: 
        Optional arguments passed to `numpy.pad`_.

    Shape
    -----
    Input: :math:`N, C_{in}, L_{in})` or :math:`(C_{in}, L_{in}`
    Output: :math:`N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out}` 
        .. math::
          L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} 
                    - \text{dilation}   \times (\text{kernel_size} - 1) - 1}
                    {\text{stride}} + 1\right\rfloor

    Attributes
    ----------
    w: Tensor 
        The learnable weights of the module of shape 
        :math:`(\text{out_channels}, 
        \frac{\text{in_channels}}{\text{groups}}, \text{kernel_size})`.
        The values of these weights are sampled from
        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
        :math:`k = \frac{groups}{C_\text{in} * \text{kernel_size}}`
    b: Tensor   
        The learnable bias of the module of shape
        (out_channels). If :attr:`bias` is ``True``, then the values of 
        these weights are sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 
        where :math:`k = \frac{groups}{C_\text{in} * \text{kernel_size}}`

    Examples
    --------
    >>> m = nn.Conv1D(33, 3, stride=2)
    >>> input = Tensor.empty(20, 16, 50).uniform()
    >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _PyTorch Conv1d:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html

    .. _Animated AI:
        https://www.youtube.com/watch?v=vVaRhZXovbw

    .. _PyTorch LazyConv1d:
        https://pytorch.org/docs/stable/generated/torch.nn.LazyConv1d.html

    .. _numpy.pad: 
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """ 
    _valid_input_dims = (2, 3)
    _error_message  = "Conv1D only accepts input tensors of shapes:\n"
    _error_message += "\t(N, C_in, W_in) or (C_in, W_in)\n"
    _error_message += "\tgot: {shape}"
    _conv_dims = 1
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Conv2D(ConvND):
    r"""
    2D convolution layer.
    
    Adapted from `PyTorch Conv2d`_. Note that this implementation 
    matches `PyTorch LazyConv2d`_, so **in_channels** is inferred
    at runtime.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    * :attr:`stride` 
        Controls the stride for the cross-correlation, a single number 
        or a one-element tuple.

    * :attr:`padding` 
        Controls the amount of padding applied to the input. It can be 
        either a string {'same'}, a tuple of ints or tuples of two 
        ints (at the same time) giving the amount of implicit padding 
        applied on both sides. Internally calls :meth:`giagrad.Tensor.pad`.

    * :attr:`dilation` 
        Controls the spacing between the kernel points; also known as 
        the à trous algorithm. It is harder to describe, but this 
        `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` 
        Controls the connections between inputs and outputs. 
        ``in_channels`` and :attr:`out_channels` must both be divisible 
        by :attr:`groups`. For example,
        
        - At groups = 1
            all inputs are convolved to all outputs.

        - At groups = 2 
            the operation becomes equivalent to having two 
            conv layers side by side, each seeing half the input channels
            and producing half the output channels, and both subsequently
            concatenated.

        - At groups = ``in_channels`` 
            each input channel is convolved 
            with its own set of filters (of size 
            :math:`\frac{\text{out_channels}}{\text{in_channels}}`).
      
    See `Animated AI`_ for an outstanding explanation.

    Note
    ----
    When `groups == in_channels` and  `out_channels == K * in_channels`, 
    where `K` is a positive integer, this operation is also known as a 
    "depthwise convolution".

    In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
    a depthwise convolution with a depthwise multiplier `K` can be 
    performed with the arguments 
    :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`.
    See `Animated AI`_.

    Note
    ----
    ``padding='valid'`` is the same as padding=0, which is the 
    default value. However ``padding='same'`` pads the input so the 
    output has the shape as the input. However, in some cases 
    padding before and after may vary, mainly due tue assymetrical 
    kernel sizes. For example, the formula to compute padding for a 
    given dimension :math:`X_{in}`:

    .. math::
        \begin{align*}
            padding &= \frac{
                (X_{in}-1) \times \text{stride} - X_{in} + K_{X_{in}} 
                + (K_{X_{in}}-1)*(\text{dilation}-1)
            }{2} \\
            pad_{before} &= \lceil padding \rceil  \\
            pad_{after}  &= \lfloor padding \rfloor  \\
        \end{align*}
                                               
    Parameters
    ----------
    out_channels: int
        Number of channels produced by the convolution.
    kernel_size: int or Tuple[int, int]
        Size of the convolving kernel.
    stride: int or Tuple[int, int], default: 1 
        Stride of the convolution. 
    dilation: int or Tuple[int, int], default: 1 
        Spacing between kernel elements. 
    padding: int, tuple or str, default: 0 
        Padding added to both sides of the input. See 
        :meth:`giagrad.Tensor.pad`.
    padding_mode: str, default: 'constant'
        Padding mode defined by `numpy.pad`_.
    groups: int, default: 1 
        Number of blocked connections from input channels to output 
        channels.
    bias: bool, default: ``True``
        If ``True``, adds a learnable bias to the output.
    **padding_kwargs: 
        Optional arguments passed to `numpy.pad`_.

    Shape
    -----
    Input: :math:`N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in}`
    Output: :math:`N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out}` 
        .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

        .. math::
          W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                    \times (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes
    ----------
    w: Tensor 
        The learnable weights of the module of shape
        :math:`(\text{out_channels}, \frac{\text{in_channels}}{\text{groups}},`
        :math:`\text{kernel_size[0]}, \text{kernel_size[1]})`.
        The values of these weights are sampled from
        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
        :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel_size}[i]}`
    b: Tensor   
        The learnable bias of the module of shape
        (out_channels). If :attr:`bias` is ``True``,
        then the values of these weights are
        sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
        :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel_size}[i]}`

    Examples
    --------
    >>> # With square kernels and equal stride
    >>> m = nn.Conv2D(33, 3, stride=2)
    >>> # non-square kernels and unequal stride and with padding
    >>> m = nn.Conv2D(33, (3, 5), stride=(2, 1), padding=(4, 2))
    >>> # non-square kernels and unequal stride and with padding and dilation
    >>> m = nn.Conv2D(33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
    >>> input = Tensor.empty(20, 16, 50, 100).uniform()
    >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _PyTorch Conv2d:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    .. _Animated AI:
        https://www.youtube.com/watch?v=vVaRhZXovbw

    .. _PyTorch LazyConv2d:
        https://pytorch.org/docs/stable/generated/torch.nn.LazyConv2d.html

    .. _numpy.pad: 
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """ 
    _valid_input_dims = (3, 4)
    _error_message  = "Conv2D only accepts input tensors of shapes:\n"
    _error_message += "\t(N, C_in, H_in, W_in) or (C_in, H_in, W_in)\n"
    _error_message += "\tgot: {shape}"
    _conv_dims = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Conv3D(ConvND):
    r"""
    3D convolution layer.
    
    Adapted from `PyTorch Conv3d`_. Note that this implementation 
    matches `PyTorch LazyConv3d`_, so **in_channels** is inferred
    at runtime.

    In the simplest case, the output value of the layer with input 
    size :math:`(N, C_{in}, D, H, W)` and output 
    :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` can be precisely 
    described as:

    .. math::
        out(N_i, C_{out_j}) = bias(C_{out_j}) +
            \sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \star input(N_i, k)

    where :math:`\star` is the valid 3D `cross-correlation`_ operator.

    * :attr:`stride` 
        Controls the stride for the cross-correlation, a single number 
        or a one-element tuple.

    * :attr:`padding` 
        Controls the amount of padding applied to the input. It can be 
        either a string {'same'}, a tuple of ints or tuples of two 
        ints (at the same time) giving the amount of implicit padding 
        applied on both sides. Internally calls :meth:`giagrad.Tensor.pad`.

    * :attr:`dilation` 
        Controls the spacing between the kernel points; also known as 
        the à trous algorithm. It is harder to describe, but this 
        `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` 
        Controls the connections between inputs and outputs. 
        ``in_channels`` and :attr:`out_channels` must both be divisible 
        by :attr:`groups`. For example,
        
        - At groups = 1
            all inputs are convolved to all outputs.

        - At groups = 2 
            the operation becomes equivalent to having two 
            conv layers side by side, each seeing half the input channels
            and producing half the output channels, and both subsequently
            concatenated.

        - At groups = ``in_channels`` 
            each input channel is convolved 
            with its own set of filters (of size 
            :math:`\frac{\text{out_channels}}{\text{in_channels}}`).
      
    See `Animated AI`_ for an outstanding explanation.

    Note
    ----
    When `groups == in_channels` and  `out_channels == K * in_channels`, 
    where `K` is a positive integer, this operation is also known as a 
    "depthwise convolution".

    In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
    a depthwise convolution with a depthwise multiplier `K` can be 
    performed with the arguments 
    :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`.
    See `Animated AI`_.

    Note
    ----
    ``padding='valid'`` is the same as padding=0, which is the 
    default value. However ``padding='same'`` pads the input so the 
    output has the shape as the input. However, in some cases 
    padding before and after may vary, mainly due tue assymetrical 
    kernel sizes. For example, the formula to compute padding for a 
    given dimension :math:`X_{in}`:

    .. math::
        \begin{align*}
            padding &= \frac{
                (X_{in}-1) \times \text{stride} - X_{in} + K_{X_{in}} 
                + (K_{X_{in}}-1)*(\text{dilation}-1)
            }{2} \\
            pad_{before} &= \lceil padding \rceil  \\
            pad_{after}  &= \lfloor padding \rfloor  \\
        \end{align*}
                                               
    Parameters
    ----------
    out_channels: int
        Number of channels produced by the convolution.
    kernel_size: int or Tuple[int, int, int]
        Size of the convolving kernel.
    stride: int or Tuple[int, int, int], default: 1 
        Stride of the convolution. 
    dilation: int or Tuple[int, int, int], default: 1 
        Spacing between kernel elements. 
    padding: int, tuple or str, default: 0 
        Padding added to both sides of the input. See 
        :meth:`giagrad.Tensor.pad`.
    padding_mode: str, default: 'constant'
        Padding mode defined by `numpy.pad`_.
    groups: int, default: 1 
        Number of blocked connections from input channels to output 
        channels.
    bias: bool, default: ``True``
        If ``True``, adds a learnable bias to the output.
    **padding_kwargs: 
        Optional arguments passed to `numpy.pad`_.

    Shape
    -----
    Input: :math:`N, C_{in}, D_{in}, H_{in}, W_{in})` or :math:`(C_{in}, D_{in}, H_{in}, W_{in}`
    Output: :math:`N, C_{out}, D_{out}, H_{out}, W_{out})` or :math:`(C_{out}, D_{out}, H_{out}, W_{out}`,
        .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
                    \times (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

        .. math::
            H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
                  \times (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
        
        .. math::
            W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2]
                  \times (\text{kernel_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Attributes
    ----------
    w: Tensor 
        The learnable weights of the module of shape
        :math:`(\text{out_channels}, \frac{\text{in_channels}}{\text{groups}},`
        :math:`\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`.
        The values of these weights are sampled from
        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
        :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel_size}[i]}`
    b: Tensor   
        The learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
        then the values of these weights are
        sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
        :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel_size}[i]}`

    Examples
    --------
    >>> # With square kernels and equal stride
    >>> m = nn.Conv3d(33, 3, stride=2)
    >>> # non-square kernels and unequal stride and with padding
    >>> m = nn.Conv3d(33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
    >>> input = Tensor.empty(20, 16, 10, 50, 100).uniform()
    >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _PyTorch Conv3d:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html

    .. _Animated AI:
        https://www.youtube.com/watch?v=vVaRhZXovbw

    .. _PyTorch LazyConv3d:
        https://pytorch.org/docs/stable/generated/torch.nn.LazyConv3d.html

    .. _numpy.pad: 
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """ 
    _valid_input_dims = (4, 5)
    _error_message  = "Conv3D only accepts input tensors of shapes:\n"
    _error_message += "\t(N, C_in, D_in, H_in, W_in) or (C_in, D_in, H_in, W_in)\n"
    _error_message += "\tgot: {shape}"
    _conv_dims = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
