# mathops : MATH OPeratorS
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple
from giagrad.tensor import Context

# ***** math functions (binary) ******
class Add(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, t2) -> Tuple[NDArray, Add]:
        return t1.data + t2.data, cls(t1, t2)

    def backward(self, grad_output: NDArray):
        p1, p2 = self.parents
        if p1.requires_grad:
            p1.grad += grad_output  

        if p2.requires_grad:
            p2.grad += grad_output   

    def __str__(self):
        return '+'

class Sub(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, t2) -> Tuple[NDArray, Sub]:
        return t1.data - t2.data, cls(t1, t2)

    def backward(self, grad_output: NDArray):
        p1, p2 = self.parents
        if p1.requires_grad:
            p1.grad += grad_output  

        if p2.requires_grad:
            p2.grad -= grad_output 

    def __str__(self):
        return '-'

class Mul(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, t2) -> Tuple[NDArray, Mul]:
        return t1.data * t2.data, cls(t1, t2)

    def backward(self, grad_output: NDArray):
        p1, p2 = self.parents
        if p1.requires_grad:
            p1.grad += grad_output * p2.data

        if p2.requires_grad:
            p2.grad += grad_output * p1.data

    def __str__(self):
        return '*'

class Matmul(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, t2) -> Tuple[NDArray, Matmul]:
        return t1.data.dot(t2.data), cls(t1, t2)

    def backward(self, grad_output: NDArray):
        p1, p2 = self.parents
        if p1.requires_grad:
            p1.grad += grad_output.dot(p2.data.T)

        if p2.requires_grad:
            p2.grad += p1.data.T.dot(grad_output)  

    def __str__(self):
        return 'dot'

# ***** math functions (unary) ******
class Pow(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, t2) -> Tuple[NDArray, Pow]:
        return t1.data ** t2.data, cls(t1, t2)

    def backward(self, grad_output: NDArray):
        p1, p2 = self.parents
        if p1.requires_grad:
            p1.grad += grad_output * (p2.data * p1.data)

    def __str__(self):
        return '**'

class Exp(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, Exp]:
        return np.exp(t1.data), cls(t1)

    def backward(self, grad_output: NDArray):
        p1 = self.parents[0]
        if p1.requires_grad:
            p1.grad += grad_output

    def __str__(self):
        return 'exp'

# TODO test Log and Reciprocal and Abs
class Log(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, Log]:
        return np.log(t1.data), cls(t1)

    def backward(self, grad_output: NDArray):
        p1 = self.parents[0]
        if p1.requires_grad:
            p1.grad += grad_output * np.reciprocal(p1.data)

    def __str__(self):
        return 'ln'

class Reciprocal(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, Reciprocal]:
        return np.reciprocal(t1.data), cls(t1)

    def backward(self, grad_output: NDArray):
        p1 = self.parents[0]
        if p1.requires_grad:
            p1.grad += grad_output * (-np.ones_like(p1.data) / (p1.data ** 2))

    def __str__(self):
        return 'reciprocal'

class Abs(Context):
    def _init_(self, *tensors):
        super()._init_(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, Abs]:
        return np.abs(t1.data), cls(t1)

    def backward(self, grad_output: NDArray):
        p1 = self.parents[0]
        if p1.requires_grad:
            p1.grad += grad_output * (p1.data * np.reciprocal(np.abs(p1.data)))
    
    def __str__(self):
        return 'abs'
