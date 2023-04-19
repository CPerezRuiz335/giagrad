from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple, TypeVar
from giagrad.tensor import Function

"""
Numpy tranpose creates a view, but we are 
interested in changing grad. From an operator
we can not do that. 
"""

class _Permute(Function):
    def __init__(self, *tensors, axis: Tuple[int]):
        super().__init__(tensors)
        self.axis = axis
        self._name += f"(axis = {self.axis})"

    @classmethod
    def forward(cls, t1, axis=None) -> Tuple[NDArray, _Permute]:
        axis = tuple(range(t1.ndim))[::-1] if axis is None else axis
        return np.transpose(t1.data, axes=axis), cls(t1, axis=axis)

    def backward(self, partial: NDArray):
        """Partial is already p.grad but not unpermuted"""
        p = self.parents[0]
        if p.requires_grad:
            p.grad += np.transpose(partial, np.argsort(self.axis))