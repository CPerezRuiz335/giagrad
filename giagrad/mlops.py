# mlops : Macine Learning  OPeratorS
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple
from giagrad.tensor import Context


# ***** activation functions *****
class ReLU(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, ReLU]:
        return np.maximum(t1.data, 0), cls(t1) 

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            print('partial', partial)
            p.grad += partial * (p.data > 0).astype(int)
            print('p.grad', p.grad)

    def __str__(self):
        return f"ReLU"

class ReLU6(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, ReLU6]:
        return np.minimum(np.maximum(t1.data, 0), 6), cls(t1) 

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad += partial * np.logical_and(6 > p.data, p.data > 0).astype(int)

    def __str__(self):
        return f"ReLU6"

class Hardswish(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, Hardswish]:
        return t1.data * (np.minimum(np.maximum(t1.data + 3, 0), 6) / 6), cls(t1) 

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            out = np.where(p.data < 3, (2*p.data + 3) / 6, 1) * (p.data > -3).astype(int)
            p.grad += partial * out

    def __str__(self):
        return f"Hardswish"

class Sigmoid(Context): 
    def __init__(self, *tensors, child_data: NDArray):
        self.c = child_data
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, Sigmoid]:
        out = 1 / (1 + np.exp(-t1.data))
        return out, cls(t1, child_data=out)

    def backward(self, partial: NDArray):
        p, c = self.parents[0], self.c
        if p.requires_grad:
            p.grad += partial * (c * (1 - c))

    def __str__(self):
        return "Sigmoid"

class ELU(Context):
    def __init__(self, *tensors, alpha: float):
        self.alpha = alpha
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, alpha: float) -> Tuple[NDArray, ELU]:
        return np.where(
            t1.data > 0, 
            t1.data, 
            alpha * (np.exp(t1.data) - 1)
        ), cls(t1, alpha=alpha)

    def backward(self, partial: NDArray):
        p, alpha = self.parents[0], self.alpha
        if p.requires_grad:
            p.grad += partial \
                    * np.where(
                        p.data > 0, 
                        np.ones_like(p.data), 
                        alpha * np.exp(p.data)
                    )

    def __str__(self):
        return f"ELU(alpha={self.alpha})"

class SiLU(Context):
    def __init__(self, *tensors, child_data: NDArray, beta: float):
        self.c = child_data
        self.beta = beta
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, beta: float) -> Tuple[NDArray, SiLU]:
        out = t1.data / (1 + np.exp(-beta * t1.data))
        return out, cls(t1, child_data=out, beta=beta)

    def backward(self, partial: NDArray):
        p, c, beta = self.parents[0], self.c, self.beta
        if p.requires_grad:
            out = (beta*c + 1/(1 + np.exp(-beta * p.data)) * (1 - beta*c)) 
            p.grad += partial * out

    def __str__(self):
        return f"SiLU(beta={self.beta})" if self.beta != 1.702 else "QuickGELU"


class Tanh(Context):
    def __init__(self, *tensors, child_data: NDArray):
        self.c = child_data
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, Tanh]:
        out = np.tanh(t1.data)
        return out, cls(t1, child_data=out)

    def backward(self, partial: NDArray):
        p, c = self.parents[0], self.c
        if p.requires_grad:
            p.grad += partial * (1 - c)

    def __str__(self):
        return "tanh"

class LeakyReLU(Context):
    def __init__(self, *tensors, neg_slope: float):
        self.neg_slope = neg_slope
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, neg_slope: float) -> Tuple[NDArray, LeakyReLU]:
        return np.where(
            t1.data > 0, 
            t1.data, 
            t1.data*neg_slope
        ), cls(t1, neg_slope=neg_slope)

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad += partial * np.where(p.data > 0, 1, self.neg_slope)

    def __str__(self):
        return f"LeakyReLU(neg_slope={self.neg_slope})"


class Softplus(Context):
    def __init__(self, *tensors, limit: float, beta: float):
        self.limit = limit
        self.beta = beta
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, limit: float, beta: float) -> Tuple[NDArray, Softplus]:
        return np.where(
            t1.data > limit,
            t1.data,
            (1/beta) * np.log(1 + np.exp(beta * t1.data))
        ), cls(t1, limit=limit, beta=beta)

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad += partial \
                    * np.where(
                    p.data > self.limit,
                    1,
                    1 / (1 + np.exp(-self.beta*p.data))
                )

    def __str__(self):
        return f"Softplus(lim={self.limit}, alpha={self.alpha})"


class GELU(Context):
    """https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/neural_nets/activations/activations.py#l210"""
    def __init__(self, *tensors):
        self.erf_prime = lambda x: (2 / np.sqrt(np.pi)) * np.exp(-(x ** 2)) 
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, GELU]:
        out = 0.5 * t1.data \
            * (1 + np.tanh(np.sqrt(2 / np.pi) \
            * (t1.data + 0.044715 * t1.data ** 3)))
        return out, cls(t1)

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            out = 0.5 \
                + 0.5 * np.tanh(np.sqrt(2 / np.pi) * (p.data + 0.044715 * p.data ** 3)) \
                + ((0.5 * p.data * self.erf_prime(p.data/np.sqrt(2))) / np.sqrt(2))
            p.grad += partial * out

    def __str__(self):
        return "GELU"

class Softmax(Context):
    def __init__(self, *tensors, child_data: NDArray):
        self.c = child_data
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, Softmax]:
        assert t1.data.shape[1] == 1, "Softmax should be used with column vectors"
        tmp = np.exp(t1.data - np.max(t1.data))
        out = tmp / np.sum(tmp)
        return out, cls(t1, child_data=out)

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            n = np.size(self.c)
            p.grad += np.dot((np.identity(n) - self.c.T) * self.c, partial)

    def __str__(self):
        return "Softmax"

class LogSoftmax(Context):
    def __init__(self, *tensors, child_data: NDArray):
        self.c = child_data
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[NDArray, LogSoftmax]:
        assert t1.data.shape[1] == 1, "Softmax should be used with column vectors"
        m = np.max(t1.data)
        tmp = np.log(np.exp(t1.data - m).sum())
        out = t1.data - m - tmp
        return out, cls(t1, child_data=out)

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            n = np.size(self.c)
            p.grad += np.dot(np.identity(n) - self.c.T, partial) 

    def __str__(self):
        return "LogSoftmax"