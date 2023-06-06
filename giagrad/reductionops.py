from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple, Union, Optional, Literal, Callable
from giagrad.tensor import Function
from itertools import zip_longest
import math

def expand(partial: NDArray, p_shape: Tuple[int, ...], axis: Union[Tuple[int, ...], int, None]):
    if axis is None: return partial # if True partial is scalar-valued
    if isinstance(axis, int): axis = (axis,)
    # check if positive or negative
    axis = tuple(len(p_shape)+ax if ax < 0 else ax for ax in axis)
    newshape = tuple(1 if i in axis else ax for i, ax in enumerate(p_shape))
    return np.reshape(partial, newshape=newshape)

# **** reduction functions *****
class Sum(Function):
    def __init__(self, axis: Optional[Tuple[int, ...]], keepdims: bool):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
        self._name += f'(axis={self.axis})' if self.axis is not None else ''

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        return t1.data.sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad += expand(partial, p.shape, self.axis) * np.ones_like(p.data)


"""
Pytorch max and min reductions don't accept multiple dimensions/axis, just int.
However, with giagrad max and min reduction operators it could be technically possible.
"""
class MinMax(Function):
    def __init__(self, axis: Optional[int], keepdims: bool, fn: Callable):
        super().__init__()
        self.fn = fn
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, t1) ->NDArray:
        self.save_for_backward(t1)
        # fn is either np.max or np.min
        # d max/ dx when there are ties is undefined, avg of ties instead
        self.minmax = self.fn(t1.data, axis=self.axis, keepdims=self.keepdims)
        return self.minmax

    def backward(self, partial: NDArray):
        p, minmax = self.parents[0], self.minmax
        if p.requires_grad:
            mask = (p.data == expand(minmax, p.shape, self.axis)).astype(int)
            p.grad += expand(partial, p.shape, self.axis) * (mask / mask.sum(axis=self.axis, keepdims=True))

    def __str__(self):
        fn_str = "Min" if self.fn is np.min else "Max"
        axis = "()" if self.axis is None else f"(axis={self.axis})"
        return f'{fn_str}{axis}' if not self._name else self._name   


class Mean(Function):
    def __init__(self, axis: Optional[Tuple[int, ...]], keepdims: bool):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
        self._name += f'(axis = {self.axis})' if self.axis is not None else ''

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        return t1.data.mean(axis=self.axis, keepdims=self.keepdims)

    def __constant(self) -> float:
        p = self.parents[0]
        if self.axis is None: 
            return 1 / p.size
        elif isinstance(self.axis, int): 
            return 1 / p.shape[self.axis]
        return 1 / math.prod(p.shape[i] for i in self.axis)

    def backward(self, partial: NDArray):
        p = self.parents[0]
        prob = self.__constant()
        if p.requires_grad:
            p.grad += expand(partial, p.shape, self.axis) * np.full_like(p.data, prob)