# reductionops : REDUCTION OPeratorS
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple, Union, Optional
from giagrad.tensor import Context
from itertools import zip_longest
import math

def expand(partial: NDArray, p_shape: Tuple[int, ...], axis: Union[Tuple[int, ...], int, None]):
    # if True => reduction operator was called with keepdims = False
    if axis is None: return partial
    if isinstance(axis, int): axis = (axis,)
    newshape = (1 if i in axis else ax for i, ax in enumerate(p_shape))
    return np.reshape(partial, newshape=tuple(newshape))

# **** reduction functions *****
class Sum(Context):
    def __init__(self, *tensors, axis: Optional[Tuple[int, ...]]):
        self.axis = axis
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, axis: Optional[Tuple[int, ...]], keepdims: bool) -> Tuple[Union[NDArray, float], Sum]:
        return t1.data.sum(axis=axis, keepdims=keepdims), cls(t1, axis=axis)

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad += expand(partial, p.shape, self.axis) * np.ones_like(p.data)

    def __str__(self):
        return f'Sum(axis = {self.axis})'           

"""
Pytorch max and min reductions don't accept multiple dimensions/axis, just int.
However, with giagrad max and min reduction operators it could be tecnically possible
but may not be useful or correct.
"""

class Max(Context):
    def __init__(self, *tensors, max_: Union[NDArray, float], axis = Optional[int]):
        self.max_ = max_
        self.axis = axis
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, axis: Optional[int], keepdims: bool) -> Tuple[Union[NDArray, float], Max]:
        """d max/ dx when there are ties is undefined, avg of ties instead"""
        max_ = t1.data.max(axis=axis, keepdims=keepdims)
        return max_, cls(t1, max_=max_, axis=axis)

    def backward(self, partial: NDArray):
        p, max_ = self.parents[0], self.max_
        if p.requires_grad:
            mask = (p.data == expand(max_, p.shape, self.axis)).astype(int)
            p.grad += expand(partial, p.shape, self.axis) \
                      * (mask / mask.sum(axis=self.axis, keepdims=True))

    def __str__(self):
        return f'Max(axis = {self.axis})'   


class Min(Context):
    def __init__(self, *tensors, min_: Union[NDArray, float], axis: Optional[int]):
        self.min_ = min_
        self.axis = axis
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, axis: Optional[int], keepdims: bool) -> Tuple[Union[NDArray, float], Min]:
        """d min/ dx when there are ties is undefined, avg of ties instead"""
        min_ = t1.data.min(axis=axis, keepdims=keepdims)
        return min_, cls(t1, min_=min_, axis=axis)

    def backward(self, partial: NDArray):
        p, min_ = self.parents[0], self.min_
        if p.requires_grad:
            mask = (p.data == expand(min_, p.shape, self.axis)).astype(int)
            p.grad += expand(partial, p.shape, self.axis) \
                      * (mask / mask.sum(axis=self.axis, keepdims=True))

    def __str__(self):
        return f'Min(axis = {self.axis})'  

class Mean(Context):
    def __init__(self, *tensors, axis: Optional[Tuple[int, ...]]):
        self.axis = axis
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, axis: Optional[Tuple[int, ...]], keepdims: bool) -> Tuple[Union[NDArray, float], Mean]:
        return t1.data.mean(axis=axis, keepdims=keepdims) , cls(t1, axis=axis)

    def __constant(self) -> float:
        p = self.parents[0]
        p_shape = p.shape 

        if self.axis is None: 
            prob = 1 / math.prod(p_shape)
        elif isinstance(self.axis, int): 
            prob = 1 / p_shape[self.axis]
        else:
            prob = 1 / math.prod(p_shape[i] for i in self.axis)

        return prob

    def backward(self, partial: NDArray):
        p = self.parents[0]
        prob = self.__constant()
        if p.requires_grad:
            p.grad +=  expand(partial, p.shape, self.axis) * np.full_like(p.data, prob)

    def __str__(self):
        return f'Mean(axis = {self.axis})'    