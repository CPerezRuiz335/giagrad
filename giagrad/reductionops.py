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
        return t1.data.sum(), cls(t1)

    def backward(self, grad_output: NDArray):
        p1 = self.parents[0]
        if p1.requires_grad:
            new_grad = grad_output * np.ones_like(p1.data)
            p1.grad = new_grad if p1.grad is None else (p1.grad + new_grad)

    def __str__(self):
        return 'sum'           

# TODO, falla, los de reduccion modifican el gradiente de lo que sale, no solo de
# lo que entra

class Max(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, Max]:
        return t1.data.max(), cls(t1)

    def backward(self, grad_output: NDArray):
        p1 = self.parents[0]
        if p1.requires_grad:
            new_grad = grad_output * np.ones_like(p1.data)
            p1.grad = new_grad if p1.grad is None else (p1.grad + new_grad)

    def __str__(self):
        return 'max'   


class Min(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, Min]:
        return t1.data.min(), cls(t1)

    def backward(self, grad_output: NDArray):
        p1 = self.parents[0]
        if p1.requires_grad:
            new_grad = grad_output * np.ones_like(p1.data)
            p1.grad =  new_grad if p1.grad is None else (p1.grad + new_grad)

    def __str__(self):
        return 'min'  

class Mean(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, Mean]:
        return t1.data.mean(), cls(t1)

    def backward(self, grad_output: NDArray):
        p1 = self.parents[0]
        if p1.requires_grad:
            new_grad = grad_output * np.ones_like(p1.data)
            p1.grad =  new_grad if p1.grad is None else (p1.grad + new_grad)

    def __str__(self):
        return 'mean'    