from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple, TypeVar
from giagrad.tensor import Context

"""
Numpy tranpose creates a view, but we are 
interested in changing grad. From an operator
we can not do that. 
"""

class Permute(Context):
    def __init__(self, *tensors, axes: Tuple[int]):
        self.axes = axes
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, axes=None) -> Tuple[NDArray, Permute]:
        axes = tuple(range(t1.ndim))[::-1] if axes is None else axes
        return np.transpose(t1.data, axes=axes), cls(t1, axes=axes)

    def backward(self, partial: NDArray):
        """Partial is already p.grad but not unpermuted"""
        p = self.parents[0]
        if p.requires_grad:
            p.grad += np.transpose(partial, np.argsort(self.axes))

    def __str__(self):
        return f"Permute {self.axes}"

