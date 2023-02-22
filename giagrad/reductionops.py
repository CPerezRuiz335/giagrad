# reductionops : REDUCTION OPeratorS
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple, Union
from giagrad.tensor import Context
import math

class Reduction:
    """
    PyTorch forces gradient to hace the same shape as
    data, which is a wise decision, but sometimes may
    not be mathematically rigorous. Thus reduction 
    operators need to be disintguished from the rest.

    That reshaping process only consists in reducing
    like sum(), see Torch.backward() and tests/operationstTests.ipynb
    """
    ...

# **** reduction functions *****
class Sum(Context, Reduction):
    def __init__(self, *tensors):
        super(Sum, self).__init__(tensors)

    @classmethod
    def forward(cls, t1, axis=1) -> Tuple[float, Sum]:
        return t1.data.sum(), cls(t1)

    def backward(self, partial: NDArray):
        assert partial.shape == (), "partial needs to be a scalar"
        p1 = self.parents[0]
        if p1.requires_grad:
            p1.grad += partial * np.ones_like(p1.data)

    def __str__(self):
        return 'sum'           

class Max(Context, Reduction):
    def __init__(self, *tensors):
        super(Max, self).__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[float, Max]:
        """d max/ dx when there are ties is undefined, avg of ties instead"""
        mmax = t1.data.max()
        mask = (t1.data == mmax) 
        mask = mask / mask.sum()
        return mmax, cls(t1, mask)

    def backward(self, partial: NDArray):
        assert partial.shape == (), "partial needs to be a scalar"
        p1, mask = self.parents
        if p1.requires_grad:
            p1.grad += partial * mask

    def __str__(self):
        return 'max'   


class Min(Context, Reduction):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[float, Min]:
        """d min/ dx when there are ties is undefined, avg of ties instead"""
        minn = t1.data.min()
        mask = (t1.data == minn) 
        mask = mask / mask.sum()
        return minn, cls(t1, mask)

    def backward(self, partial: NDArray):
        assert partial.shape == (), "partial needs to be a scalar"
        p1, mask = self.parents
        if p1.requires_grad:
            p1.grad +=  partial * mask

    def __str__(self):
        return 'min'  

class Mean(Context, Reduction):
    def __init__(self, *tensors):
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1) -> Tuple[float, Mean]:
        mask = np.ones_like(t1.data) / math.prod(t1.shape)
        return t1.data.mean(), cls(t1)

    def backward(self, partial: NDArray):
        assert partial.shape == (), "partial needs to be a scalar"
        p1, mask = self.parents
        if p1.requires_grad:
            p1.grad +=  partial * mask

    def __str__(self):
        return 'mean'    