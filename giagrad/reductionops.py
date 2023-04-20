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
class _Sum(Function):
    def __init__(self, *tensors, axis: Optional[Tuple[int, ...]]):
        super().__init__(tensors)
        self.axis = axis
        self._name += f'(axis={self.axis})' if self.axis is not None else ''

    @classmethod
    def forward(cls, t1, axis: Optional[Tuple[int, ...]], keepdims: bool) -> Tuple[Union[NDArray, float], _Sum]:
        return t1.data.sum(axis=axis, keepdims=keepdims), cls(t1, axis=axis)

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad += expand(partial, p.shape, self.axis) * np.ones_like(p.data)


"""
Pytorch max and min reductions don't accept multiple dimensions/axis, just int.
However, with giagrad max and min reduction operators it could be technically possible.
"""
class _MinMax(Function):
    def __init__(self, *tensors, minmax: Union[NDArray, float], axis = Optional[int], fn: Callable):
        super().__init__(tensors)
        self.fn = fn
        self.minmax = minmax
        self.axis = axis

    @classmethod
    def forward(cls, t1, axis: Optional[int], keepdims: bool, fn: Callable) -> Tuple[Union[NDArray, float], _MinMax]:
        # fn is either np.max or np.min
        # d max/ dx when there are ties is undefined, avg of ties instead
        minmax = fn(t1.data, axis=axis, keepdims=keepdims)
        return minmax, cls(t1, minmax=minmax, axis=axis, fn=fn)

    def backward(self, partial: NDArray):
        p, minmax = self.parents[0], self.minmax
        if p.requires_grad:
            mask = (p.data == expand(minmax, p.shape, self.axis)).astype(int)
            p.grad += expand(partial, p.shape, self.axis) * (mask / mask.sum(axis=self.axis, keepdims=True))

    def __str__(self):
        fn_str = "Min" if self.fn is np.min else "Max"
        axis = "()" if self.axis is None else f"(axis={self.axis})"
        return f'{fn_str}{axis}' if not self._name else self._name   


class _Mean(Function):
    def __init__(self, *tensors, axis: Optional[Tuple[int, ...]]):
        super().__init__(tensors)
        self.axis = axis
        self._name += f'(axis = {self.axis})' if self.axis is not None else ''

    @classmethod
    def forward(cls, t1, axis: Optional[Tuple[int, ...]], keepdims: bool) -> Tuple[Union[NDArray, float], _Mean]:
        return t1.data.mean(axis=axis, keepdims=keepdims) , cls(t1, axis=axis)

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