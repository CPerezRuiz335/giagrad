# mlops : Macine Learning  OPeratorS
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple
from giagrad.tensor import Function

def stable_sigmoid(data: NDArray):
    return np.exp(-np.logaddexp(0.0, -data))

# ***** activation functions *****
class _ReLU(Function):
    def __init__(self):
        super().__init__()

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        return np.maximum(t1.data, 0) 

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad += partial * (p.data > 0).astype(int)

class _ReLU6(Function):
    def __init__(self):
        super().__init__()

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        return np.minimum(np.maximum(t1.data, 0), 6)

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad += partial * np.logical_and(6 > p.data, p.data > 0).astype(int)

class _Hardswish(Function):
    def __init__(self):
        super().__init__()

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        return t1.data * (np.minimum(np.maximum(t1.data + 3, 0), 6) / 6)

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            out = np.where(p.data < 3, (2*p.data + 3) / 6, 1) * (p.data > -3).astype(int)
            p.grad += partial * out

class _Sigmoid(Function): 
    def __init__(self):
        super().__init__()
    
    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        # save sigmoid output for backprop
        self.s = stable_sigmoid(t1.data)
        return self.s

    def backward(self, partial: NDArray):
        p, s = self.parents[0], self.s
        if p.requires_grad:
            p.grad += partial * (s * (1 - s))

class _ELU(Function):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
        self._name += f"(alpha={self.alpha})"

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        return np.where(
            t1.data > 0, 
            t1.data, 
            self.alpha * (np.exp(t1.data) - 1)
        )

    def backward(self, partial: NDArray):
        p, alpha = self.parents[0], self.alpha
        if p.requires_grad:
            p.grad += partial \
                    * np.where(
                        p.data > 0, 
                        np.ones_like(p.data), 
                        alpha * np.exp(p.data)
                    )

class _SiLU(Function):
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta
        self._name = f"SiLU(beta={self.beta})" if self.beta != 1.702 else "QuickGELU"

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        # save output silu for backprop
        self.s = t1.data * stable_sigmoid(self.beta * t1.data)
        return self.s

    def backward(self, partial: NDArray):
        p, s, beta = self.parents[0], self.s, self.beta
        if p.requires_grad:
            out = (beta*s + 1/(1 + np.exp(-beta * p.data)) * (1 - beta*s)) 
            p.grad += partial * out

class _Tanh(Function):
    def __init__(self):
        super().__init__()
    
    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        # save tanh output for backprop
        self.s = np.tanh(t1.data)
        return self.s

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad += partial * (1 - self.s**2)

class _LeakyReLU(Function):
    def __init__(self, neg_slope: float):
        super().__init__()
        self.neg_slope = neg_slope
        self._name += f"(neg_slope={self.neg_slope})"

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        return np.where(
            t1.data > 0, 
            t1.data, 
            t1.data*self.neg_slope
        )

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad += partial * np.where(p.data > 0, 1, self.neg_slope)

class _Softplus(Function):
    def __init__(self, beta: float, limit: float):
        super().__init__()
        self.limit = limit
        self.beta = beta
        self._name = f"(beta={self.beta}, lim={self.limit})"

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        return np.where(
            t1.data * self.beta > self.limit,
            t1.data,
            (1/self.beta) * np.log(1 + np.exp(self.beta * t1.data))
        )

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad += partial \
                    * np.where(
                    p.data * self.beta > self.limit,
                    1,
                    stable_sigmoid(self.beta*p.data)
                )

class _Mish(Function):
    def __init__(self, beta: float, limit: float):
        super().__init__()
        self.limit = limit
        self.beta = beta
        self._name = f"(beta={self.beta}, lim={self.limit})"

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        soft = _Softplus(beta=self.beta, limit=self.limit).forward(t1)
        # save for backprop
        self.tanh_soft = np.tanh(soft)
        return t1.data * self.tanh_soft

    def backward(self, partial: NDArray):
        p, limit, beta, tanh_soft = self.parents[0], self.limit, self.beta, self.tanh_soft
        if p.requires_grad:
            out = tanh_soft + p.data * (1 - tanh_soft**2) \
                * np.where(
                    p.data * beta > limit,
                    1,
                    stable_sigmoid(beta*p.data)
                )
            p.grad += partial * out


import math
erf_prime = lambda x: (2 / np.sqrt(np.pi)) * np.exp(-(x ** 2)) 
erf = np.vectorize(math.erf)

class _GELU(Function):
    def __init__(self):
        super().__init__()
    
    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        self.s = t1.data / np.sqrt(2)
        return 0.5 * t1.data * (1 + erf(self.s))

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            out = 0.5 + 0.5 * erf(self.s) + ((0.5 * p.data * erf_prime(self.s)) / np.sqrt(2))
            p.grad += partial * out

class _Softmax(Function):
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis
        self._name = f"(axis={self.axis})"
    
    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        # Softmax input must be a vector of K real numbers
        # that's why axis and apply_along_axis required
        def fn(x: NDArray):
            tmp = np.exp(x - np.max(x))
            return tmp / np.sum(tmp)

        self.s = np.apply_along_axis(fn, self.axis, t1.data)
        return self.s 

    def backward(self, partial: NDArray):
        def fn(x: NDArray) -> NDArray:
            data, partial = np.split(x, 2)
            n = np.size(data)
            return np.dot(
                (np.identity(n) - data) * data[..., None], partial[..., None]
            ).flatten()

        p = self.parents[0]
        if p.requires_grad:
            p.grad += np.apply_along_axis(
                fn, 
                self.axis, 
                np.append(self.s, partial, self.axis)
            )

class _LogSoftmax(Function):
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis
        self._name = f"(axis={self.axis})"
    
    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        # LogSoftmax input must be a vector of K real numbers
        # that's why axis and apply_along_axis required
        def fn(x: NDArray) -> NDArray:
            tmp = x - np.max(x)
            return tmp - np.log(np.exp(tmp).sum())

        return np.apply_along_axis(fn, self.axis, t1.data)

    def backward(self, partial: NDArray):
        def fn(x: NDArray) -> NDArray:
            data, partial = np.split(x, 2)
            n = np.size(data)
            return np.dot(
                np.identity(n) - data[..., None], partial[..., None]
            ).flatten()

        p = self.parents[0]
        if p.requires_grad:
            # Derivative of LogSoftmax uses Softmax
            soft = _Softmax(self.axis).forward(p)
            p.grad += np.apply_along_axis(
                fn, 
                self.axis, 
                np.append(soft, partial, self.axis)
            )