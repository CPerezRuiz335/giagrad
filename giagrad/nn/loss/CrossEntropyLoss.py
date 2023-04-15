from __future__ import annotations
from giagrad.tensor import Tensor, Context
from giagrad.mlops import LogSoftmax
from giagrad.reductionops import Sum
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, Tuple
import math

class CrossEntropy(Context):
    def __init__(self, *tensor, one_hot: NDArray, log_softmax: NDArray):
        self.one_hot = one_hot
        self.softmax = np.exp(log_softmax)
        super().__init__(tensor)

    @classmethod
    def forward(cls, t1, y: NDArray, axis: int) -> Tuple[NDArray, CrossEntropy]:
        log_softmax, _ = LogSoftmax.forward(t1, axis=axis)  
        # one hot
        one_hot = np.zeros_like(t1.data)
        one_hot[np.arange(y.size), y] = 1

        return -one_hot * log_softmax, cls(t1, one_hot=one_hot, log_softmax=log_softmax)

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad += partial * (self.softmax - self.one_hot)

    def __str__(self):
        return f"CrossEntropy"

class CrossEntropyLoss:
    r"""Computes the cross entropy loss between input logits and target.

    It is useful when training a classification problem with `C` classes. The `target` is 
    expected to contain the unnormalized logits for each class (which do `not` need to be 
    positive or sum to 1, in general). `target` has to be a Tensor of size :math:`(C)` or
    :math:`(N, C)`. 

    The `target` that this criterion expects should contain indices in the range :math:`[0, C)` 
    where :math:`C` is the number of classes. :attr:`reduction` can either be ``'mean'`` (default) 
    or ``'sum'``:

    .. math::
      \ell(x, y) = \begin{cases}
          \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore_index}\}} l_n, &
           \text{if reduction} = \text{`mean';}\\
            \sum_{n=1}^N l_n,  &
            \text{if reduction} = \text{`sum'.}
        \end{cases}
      

    where :math:`l_n` is:   
    
    .. math::
      \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
      l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
      \cdot \mathbb{1}\{y_n \not= \text{ignore_index}\}

    and :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
    :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension.

    Attributes
    ----------
    reduction: str, default: 'mean'
        Specifies the reduction applied to the output: ``'mean'`` | ``'sum'``.

    Parameters
    ----------
    pred: Tensor
        Unnormalized logits.
    target: Tensor or array_like
        True labels.

    Examples
    --------
    >>> loss = nn.CrossEntropyLoss()
    >>> input = Tensor.empty(3, 5, requires_grad=True).uniform()
    >>> target = Tensor.empty(3, dtype=np.int8).uniform(b=5)
    >>> output = loss(input, target)
    >>> output.backward()
    """
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, pred: Tensor, target: Union[Tensor, NDArray]) -> Tensor:
        # weights are unnormalized logits
        # y must be a sequence of classes numerically encoded from 0 to C-1
        target = target.data if isinstance(target, Tensor) else target
        if self.reduction == 'sum':
            t = Tensor.comm(CrossEntropy, pred, y=target, axis=1).sum()
        if self.reduction == 'mean':
            t = Tensor.comm(CrossEntropy, pred, y=target, axis=1).mean(axis=0).sum()
        t._ctx._name = f"CrossEntropyLoss(reduction = {self.reduction})"
        return t    