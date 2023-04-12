from giagrad.tensor import Tensor
from giagrad.nn.module import Module
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import math

def _random_dims_to_zero(r: NDArray, p: float):
    n = math.prod(r.shape[:2]) if r.ndim > 2 else r.shape[0]
    rng = np.random.default_rng()

    for _ in range(n):
        if rng.random() > p:
            if r.ndim == 2:
                ij = rng.integers(0, r.shape[0]) 
            else:
                ij = rng.integers((0, 0), r.shape[:2]) 
            r[tuple(ij)] *= 0


class Dropout(Module):
    r"""
    Randomly zeroes some of the elements of the input tensor with probability :math:`p`.

    During training, each element of the input tensor is zeroed with probability :math:`p`,
    and the remaining elements are scaled up by a factor of :math:`\frac{1}{1-p}` to preserve the
    expected value of the output. During inference, the dropout layer does not modify
    the input tensor.

    Parameters
    ----------
    p : float, default: 0.5
        Probability of an element to be zeroed. 

    Examples
    --------
    >>> from giagrad import Tensor                                                                                       
    >>> import giagrad.nn as nn                                                                                          
    >>> dropout = nn.Dropout(p=0.3)                                                                                        
    >>> x = Tensor.empty(2, 4).ones()   
    tensor: [[1. 1. 1. 1.]
             [1. 1. 1. 1.]]                                                         
    >>> dropout(x)                                                                                           
    tensor: [[0.        0.        1.4285715 0.       ]
             [1.4285715 1.4285715 0.        0.       ]] grad_fn: Dropout
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self._name = type(self).__name__
    
    def __call__(self, x: Tensor) -> Tensor:
        r = np.random.binomial(1, self.p, size=x.shape)
        if self._train:
            x = x * r * (1 / (1 - self.p))
            x._ctx._name = self._name
            return x
        x._ctx._name = self._name
        return x

class DropoutNd(Module):
    def __init__(self):
        super().__init__()
        self.p: float
        self._name = type(self).__name__

    def __check(self, ndim: int):
        pass

    def __call__(self, x: Tensor) -> Tensor:
        self.__check(x.ndim)

        if self._train:
            r = np.ones_like(x.data)
            _random_dims_to_zero(r, self.p)
            x = x * r * (1 / (1 - self.p))
            x._ctx._name = self._name
            return x
        x._ctx._name = self._name
        return x

class Dropout1d(DropoutNd):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def __check(self, ndim: int):
        if ndim not in [2,3]:
            raise ValueError("Dropout 1d only for 2D and 3D tensors")

class Dropout2d(DropoutNd):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def __check(self, ndim: int):
        if ndim not in [3,4]:
            raise ValueError("Dropout 2d only for 3D and 4D tensors")
    
class Dropout3d(DropoutNd):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def __check(self, ndim: int):
        if ndim not in [4,5]:
            raise ValueError("Dropout 3d only for 4D and 5D tensors")
