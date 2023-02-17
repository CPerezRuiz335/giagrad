# reductionops : REDUCTION OPeratorS
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple
from giagrad.tensor import Context

# **** reduction functions *****
class Sum(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, Sum]:
        return np.sum(t1.data), cls(t1)

    def backward(self, grad_output: NDArray):
        p1 = self.parents[0]
        if p1.requires_grad:
            new_grad = grad_output * np.ones_like(p1.data)
            p1.grad =  new_grad if p1.grad is None else (p1.grad + new_grad)