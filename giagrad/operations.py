from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any
from giagrad.tensor import Tensor

class Context:

    def __init__(self, *tensors):
        assert len(tensors) <= 2, "ternary operator not supported"
        self.parents = tensors

    @classmethod
    def forward(self, *tensors, **kwargs) -> Tensor:
        """Main function for forward pass."""
        raise NotImplementedError(f"forward not implemented for {type(self)}")
    
    def backward(self, grad_output: NDArray):
        """Backprop automatic differentiation, to update grad of parents.
        grad_output: gradient of the output of forward method."""
        raise NotImplementedError(f"backward not implemented for {type(self)}")

    def __str__(self):
        """For graphviz visualization."""
        raise NotImplementedError(f"__str__ not implemented for class {type(self)}")

# ***** math functions (binary) ******

class Add(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(self, *tensors, **kwargs) -> Tensor:
        t1, t2 = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
        return Tensor(t1.data + t2.data, context=Add(t1, t2))

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
    def forward(self, *tensors, **kwargs) -> Tensor:
        t1, t2 = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
        return Tensor(t1.data - t2.data, context=Sub(t1, t2))

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
    def forward(self, *tensors, **kwargs) -> Tensor:
        t1, t2 = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
        return Tensor(t1.data * t2.data, context=Mul(t1, t2))

    def backward(self, grad_output: NDArray):
        p1, p2 = self.parents
        if p1.requires_grad:
            p1.grad += grad_output * p2.grad 
        if p2.requires_grad:
            p2.grad += grad_output * p1.grad 

    def __str__(self):
        return '*'

class Matmul(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(self, *tensors, **kwargs) -> Tensor:
        t1, t2 = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
        return Tensor(t1.data.dot(t2.data), context=Matmul(t1, t2))

    def backward(self, grad_output: NDArray):
        p1, p2 = self.parents
        if p1.requires_grad:
            p1.grad += grad_output.dot(p2.data.T)
        if p2.requires_grad:
            p2.grad += p1.T.dot(grad_output) 

    def __str__(self):
        return 'dot'

# ***** math functions (unary) ******

class Pow(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(self, tensor: Tensor, pw: int, **kwargs) -> Tensor:
        return Tensor(tensor.data ** pw, context=Pow(tensor, pw))

    def backward(self, grad_output: NDArray):
        p1, pw = self.parents
        if p1.requires_grad:
            p1.grad += grad_output * (pw * p1.data)

    def __str__(self):
        return '*'