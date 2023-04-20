from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple
from giagrad.tensor import Function
from itertools import zip_longest

def collapse(partial: NDArray, p_shape: Tuple[int, ...]):
    reduce_axis = []
    # expand dimensions 
    axes = [1 for _ in range(partial.ndim-len(p_shape))] + list(p_shape)
    # check which ones were expanded with respect to partial.shape and reduce that
    for i, ax in enumerate(zip(partial.shape, axes)):
        if ax[0] != ax[1]: reduce_axis.append(i)
    # reshape to keep parent shape 
    return np.sum(partial, axis=tuple(reduce_axis), keepdims=True).reshape(p_shape)

# ***** math functions (binary) *****
class _Add(Function):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, t2) -> Tuple[NDArray, _Add]:
        return t1.data + t2.data, cls(t1, t2)

    def backward(self, partial: NDArray):
        p1, p2 = self.parents
        if p1.requires_grad:
            p1.grad += collapse(partial, p1.grad.shape)  

        if p2.requires_grad:
            p2.grad += collapse(partial, p2.grad.shape)  

class _Sub(Function):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, t2) -> Tuple[NDArray, _Sub]:
        return t1.data - t2.data, cls(t1, t2)

    def backward(self, partial: NDArray):
        p1, p2 = self.parents
        if p1.requires_grad:
            p1.grad += collapse(partial, p1.grad.shape)    

        if p2.requires_grad:
            p2.grad -= collapse(partial, p2.grad.shape)  

class _Mul(Function):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, t2) -> Tuple[NDArray, _Mul]:
        return t1.data * t2.data, cls(t1, t2)

    def backward(self, partial: NDArray):
        p1, p2 = self.parents
        if p1.requires_grad:
            p1.grad += collapse(partial * p2.data, p1.grad.shape) 

        if p2.requires_grad:
            p2.grad += collapse(partial * p1.data, p2.grad.shape) 

class _Div(Function):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, t2) -> Tuple[NDArray, _Div]:
        return t1.data / t2.data, cls(t1, t2)

    def backward(self, partial: NDArray):
        p1, p2 = self.parents
        if p1.requires_grad:
            out = partial * (1 / p2.data)
            p1.grad += collapse(out, p1.grad.shape) 
        if p2.requires_grad:
            out = partial * (-p1.data / (p2.data**2))
            p2.grad += collapse(out, p2.grad.shape) 

class _Matmul(Function):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, t2) -> Tuple[NDArray, _Matmul]:
        return t1.data.dot(t2.data), cls(t1, t2)

    def backward(self, partial: NDArray):
        p1, p2 = self.parents
        if p1.requires_grad:
            p1.grad += partial.dot(p2.data.T)

        if p2.requires_grad:
            p2.grad += p1.data.T.dot(partial)  

# ***** math functions (unary) *****
class _Pow(Function):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, t2) -> Tuple[NDArray, _Pow]:
        return t1.data ** t2.data, cls(t1, t2)

    def backward(self, partial: NDArray):
        p1, p2 = self.parents
        if p1.requires_grad:
            p1.grad += partial * (p2.data * (p1.data ** (p2.data-1)))

class _Exp(Function):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, _Exp]:
        return np.exp(t1.data), cls(t1)

    def backward(self, partial: NDArray):
        p1 = self.parents[0]
        if p1.requires_grad:
            p1.grad += partial * np.exp(p1.data)

class _Log(Function):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, _Log]:
        return np.log(t1.data), cls(t1)

    def backward(self, partial: NDArray):
        p1 = self.parents[0]
        if p1.requires_grad:
            p1.grad += partial * np.reciprocal(p1.data)

class _Reciprocal(Function):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, _Reciprocal]:
        return np.reciprocal(t1.data), cls(t1)

    def backward(self, partial: NDArray):
        p1 = self.parents[0]
        if p1.requires_grad:
            p1.grad += partial * (-np.ones_like(p1.data) / (p1.data**2))

class _Abs(Function):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, _Abs]:
        return np.abs(t1.data), cls(t1)

    def backward(self, partial: NDArray):
        p1 = self.parents[0]
        if p1.requires_grad:
            p1.grad += partial * (p1.data / np.abs(p1.data))