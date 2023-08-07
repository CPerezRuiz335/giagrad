from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from giagrad.tensor import Tensor, Function
from giagrad.nn.containers import Module

class BatchNormND(Module):
    r"""
    Applies Batch Normalization as described in `Batch Normalization: 
    Accelerating Deep Network Training by Reducing Internal Covariate 
    Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable 
    parameter vectors of size `C` (where `C` is the number of features or
    channels). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. 
    The standard-deviation is calculated with zero degrees of freedom, 
    equivalent to :meth:`Tensor.var(ddof=0) <Tensor.var>`.

    Also by default, during training this layer keeps running estimates 
    of its computed mean and variance, which are then used for normalization 
    during evaluation. The running estimates are kept with a default 
    :attr:`momentum` of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then 
    does not keep running estimates, and batch statistics are instead 
    used during evaluation time as well.

    Note
    ----
    This :attr:`momentum` argument is different from one used in optimizer
    classes and the conventional notion of momentum. Mathematically, the
    update rule for running statistics here is
    :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
    where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is 
    the new observed value.
    

    Parameters
    ----------
    eps: default: 1e-5
        A value added to the denominator for numerical stability.
    momentum: default: 0.1
        The value used for the running_mean and running_var computation. 
        Can be set to ``None`` for cumulative moving average (i.e. simple 
        average). 
    affine: default ``True``
        A boolean value that when set to ``True``, this module has
        learnable affine parameters (:math:`\gamma` and :math:`\beta`). 
    track_running_stats: default: ``True`` 
        A boolean value that when set to ``True``, this module tracks 
        the running mean and variance, and when set to ``False``,
        this module does not track such statistics, in that case this 
        module always uses batch statistics in both training and eval 
        modes. 
    
    Shape 
    -----
    Input: :math:`N, C, *`
    Output: :math:`N, C, *` (same shape as input)

    Examples
    --------
    >>> # With Learnable Parameters
    >>> m = nn.BatchNormND()
    >>> # Without Learnable Parameters
    >>> m = nn.BatchNormND(affine=False)
    >>> t = Tensor.empty(20, 100, 35, 45).uniform()
    >>> output = m(t)
    """

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):  
        super().__init__()

        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.gamma: Tensor
        self.beta: Tensor
        self.running_mean: Tensor
        self.running_var: Tensor
        self.__initialized = False

    def __init_tensors(self, in_features: int):
        self.__initialized = True
        self.gamma = Tensor.empty(in_features, requires_grad=self.affine).ones()
        self.beta = Tensor.empty(in_features, requires_grad=self.affine).zeros()
        self.running_mean = Tensor.empty(in_features).zeros()
        self.running_var = Tensor.empty(in_features).ones()

    def __call__(self, x: Tensor) -> Tensor:
        self.__init_tensors(x.shape[1]) if not self.__initialized else ...
        axis = tuple(i for i in range(x.ndim) if i != 1)

        if self.training or (not self.track_running_stats and not self.training): 
            mean = x.mean(axis, keepdims=True)
            var = x.var(axis, ddof=0, keepdims=True)
        else:
            mean = self.running_mean.unsqueeze(axis)
            var = self.running_var.unsqueeze(axis)

        if self.training and self.track_running_stats:
            self.running_mean *= (1-self.momentum) 
            self.running_mean += (self.momentum * mean).squeeze()
            self.running_var *= (1-self.momentum) 
            self.running_var += (self.momentum * var).squeeze()
            
        gamma, beta = self.gamma.unsqueeze(axis), self.beta.unsqueeze(axis)
        return (x-mean) / (var+self.eps).sqrt() * gamma + beta

    def __str__(self):
        return (
            f"{type(self).__name__}("
            + f"eps={self.eps}, "
            + f"momentum={self.momentum}, "
            + f"affine={self.affine}"
            + (f", track_running_stats={self.track_running_stats}" 
                if not self.track_running_stats else '')
            + ')'
        )


class LayerNorm(Module):
    r"""
    Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last 
    :attr:`dimensions` dimensions. For example, if :attr:`dimensions`
    is ``2`` (a 2-dimensional shape), the mean and standard-deviation 
    are computed over the last 2 dimensions of the input tensor 
    (i.e. ``input.mean((-2, -1))``). :math:`\gamma` and :math:`\beta` 
    are learnable affine transform parameters if :attr:`elementwise_affine` 
    is ``True``. The standard-deviation is calculated with zero degrees 
    of freedom, equivalent to :meth:`Tensor.var(ddof=0) <Tensor.var>`.

    Note
    ----
    Unlike Batch Normalization and Instance Normalization, which applies
    scalar scale (:math:`\gamma`) and bias (:math:`\beta`) for each 
    entire channel/plane with the :attr:`affine` option, Layer 
    Normalization applies per-element scale and bias with 
    :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training 
    and evaluation modes.

    Parameters
    ----------
    dimensions: int 
        Last dimensions where normalization will be computed.
    eps: float, default: 1e-5
        A value added to the denominator for numerical stability. 
    elementwise_affine: boolean, default: ``True``
        A boolean value that when set to ``True``, this module has 
        learnable per-element affine parameters initialized to ones 
        (for weights) and zeros (for biases). 

    Attributes
    ----------
    gamma: 
        The learnable weights of the module of shape 
        :math:`\text{input.shape}[-dimensions:]` when 
        :attr:`elementwise_affine` is set to ``True``.
        The values are initialized to 1.
    beta:   
        The learnable weights of the module of shape 
        :math:`\text{input.shape}[-dimensions:]` when 
        :attr:`elementwise_affine` is set to ``True``.
        The values are initialized to 0.

    Shape 
    -----
    Input: :math:`N, C, *`
    Output: :math:`N, C, *` (same shape as input)

    Examples
    --------
    NLP Example

    >>> batch, sentence_length, embedding_dim = 20, 5, 10
    >>> embedding = Tensor.empty(batch, sentence_length, embedding_dim).uniform()
    >>> layer_norm = nn.LayerNorm(dimensions=1)
    >>> # Activate module
    >>> layer_norm(embedding)
    
    Image Example

    >>> N, C, H, W = 20, 5, 10, 10
    >>> input = torch.randn(N, C, H, W)
    >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    >>> # as shown in the image below
    >>> layer_norm = nn.LayerNorm(dimensions=3)
    >>> output = layer_norm(input)
    """

    def __init__(
        self,
        dimensions: int,
        eps: float = 1e-5,
        element_wise_affine: bool = True
    ):  
        super().__init__()

        self.dimensions = dimensions
        self.eps = eps
        self.element_wise_affine = element_wise_affine
        self.__initialized = False

    def __init_tensors(self, in_shape: Tuple[int, ...]):
        self.__initialized = True
        affine, shape = self.element_wise_affine, in_shape[-self.dimensions:]
        self.gamma = Tensor.empty(*shape, requires_grad=affine).ones()
        self.beta = Tensor.empty(*shape, requires_grad=affine).zeros()

    def __call__(self, x: Tensor) -> Tensor:
        self.__init_tensors(x.shape) if not self.__initialized else ...
        axis = tuple(-i for i in range(1, self.dimensions+1))
        mean = x.mean(axis, keepdims=True)
        var = x.var(axis, ddof=0, keepdims=True)
        return (x-mean) / (var+self.eps).sqrt() * self.gamma + self.beta

    def __str__(self):
        return (
            f"{type(self).__name__}("
            + f"dimensions={self.dimensions}, "
            + f"eps={self.eps}, "
            + f"element_wise_affine={self.element_wise_affine}" 
            + ')'
        )