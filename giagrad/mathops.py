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
            p1.grad = grad_output if p1.grad is None else (p1.grad + grad_output) 

        if p2.requires_grad:
            p2.grad = grad_output if p2.grad is None else (p2.grad + grad_output)   

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
            p1.grad = grad_output if p1.grad is None else (p1.grad + grad_output)  

        if p2.requires_grad:
            p2.grad = grad_output if p2.grad is None else (p2.grad - grad_output)   

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
            new_grad = grad_output * p2.data
            p1.grad = new_grad if p1.grad is None else (p1.grad + new_grad) 

        if p2.requires_grad:
            new_grad = grad_output * p1.data
            p2.grad = new_grad if p2.grad is None else (p2.grad + new_grad)

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
            new_grad = grad_output.dot(p2.data.T)
            p1.grad = new_grad if p1.grad is None else (p1.grad + new_grad)

        if p2.requires_grad:
            new_grad = p1.data.T.dot(grad_output) 
            p2.grad = new_grad if p2.grad is None else (p2.grad + new_grad) 

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
            new_grad = grad_output * (p2.data * p1.data)
            p1.grad = new_grad if p1.grad is None else (p1.grad + new_grad)

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
            p1.grad = grad_output if p1.grad is None else (p1.grad + grad_output) 

    def __str__(self):
        return 'exp'

class Log(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, Log]:
        return np.log(t1.data), cls(t1)

    def backward(self, grad_output: NDArray):
        p1 = self.parents[0]
        if p1.requires_grad:
            new_grad = grad_output * np.reciprocal(p1.data)
            p1.grad = new_grad if p1.grad is None else (p1.grad + new_grad)

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
            new_grad = grad_output * (-np.ones_like(p1.data) / (p1.data ** 2))
            p1.grad = new_grad if p1.grad is None else (p1.grad + new_grad)

    def __str__(self):
        return 'reciprocal'