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
    def __init__(self, axis: Tuple[int]):
        super().__init__()
        self.axis = axis
        self._name += f"(axis = {self.axis})"

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        self.axis = tuple(range(t1.ndim))[::-1] if self.axis is None else self.axis
        return np.transpose(t1.data, axes=self.axis)

    def backward(self, partial: NDArray):
        """Partial is already p.grad but not unpermuted"""
        p = self.parents[0]
        if p.requires_grad:
            p.grad += np.transpose(partial, np.argsort(self.axis))