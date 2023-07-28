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
    
    Shape 
    -----
    Input: :math:`(N, C, *)`
    Output: :math:`(N, C, *)` (same shape as input)

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
        axis = tuple(i for i in range(x.ndim) if i != 1)
        self.__init_tensors(x.shape[1]) if not self.__initialized else ...

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
