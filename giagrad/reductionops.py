# reductionops : REDUCTION OPeratorS
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple, Union
from giagrad.tensor import Context
import math

"""
PyTorch forces gradient to hace the same shape as
data, which is a wise decision, but sometimes may
not be mathematically rigorous. Thus reduction 
operators need to be disintguished from the rest.

That reshaping process only consists in reducing
like sum(), see Torch.backward() and tests/operationstTests.ipynb
"""

# **** reduction functions *****
class Sum(Context):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, axis=1) -> Tuple[float, Sum]:
        return t1.data.sum(), cls(t1)

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad += partial * np.ones_like(p.data)

    def __str__(self):
        return 'sum'           

class Max(Context):
    def __init__(self, *tensors, max_: float):
        self.max_ = max_
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[float, Max]:
        """d max/ dx when there are ties is undefined, avg of ties instead"""
        max_ = t1.data.max()
        return max_, cls(t1, max_=max_)

    def backward(self, partial: NDArray):
        p, max_ = self.parents[0], self.max_
        if p.requires_grad:
            mask = (p.data == max_).astype(int)
            p.grad += partial * (mask / mask.sum())

    def __str__(self):
        return 'max'   


class Min(Context):
    def __init__(self, *tensors, min_: float):
        self.min_ = min_
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[float, Min]:
        """d min/ dx when there are ties is undefined, avg of ties instead"""
        min_ = t1.data.min()
        return min_, cls(t1, min_=min_)

    def backward(self, partial: NDArray):
        p, min_ = self.parents[0], self.min_
        if p.requires_grad:
            mask = (p.data == min_).astype(int)
            p.grad +=  partial * (mask / mask.sum())

    def __str__(self):
        return 'min'  

class Mean(Context):
    def __init__(self, *tensors, prob: float):
        self.prob = prob
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[float, Mean]:
        return t1.data.mean(), cls(t1, prob=math.prod(t1.data.shape))

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad +=  partial * np.full_like(p.data, self.prob)

    def __str__(self):
        return 'mean'    