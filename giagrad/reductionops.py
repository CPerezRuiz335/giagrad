# reductionops : REDUCTION OPeratorS
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple, Union, Optional, Literal, Callable
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
        axis = "()" if self.axis is None else f"(axis = {self.axis})"
        return f'Sum{axis}' if not self._name else self._name          

"""
Pytorch max and min reductions don't accept multiple dimensions/axis, just int.
However, with giagrad max and min reduction operators it could be tecnically possible
but may not be useful or correct.
"""
class MinMax(Context):
    def __init__(self, *tensors, minmax: Union[NDArray, float], axis = Optional[int], fn: Callable):
        self.fn = fn
        self.minmax = minmax
        self.axis = axis
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, axis: Optional[int], keepdims: bool, fn: Callable) -> Tuple[Union[NDArray, float], MinMax]:
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
        axis = "()" if self.axis is None else f"(axis = {self.axis})"
        return f'{fn_str}{axis}' if not self._name else self._name   

class Mean(Context):
    def __init__(self, *tensors, axis: Optional[Tuple[int, ...]]):
        self.axis = axis
        super().__init__(tensors)

    @classmethod
    def forward(cls, t1, axis: Optional[Tuple[int, ...]], keepdims: bool) -> Tuple[Union[NDArray, float], Mean]:
        return t1.data.mean(axis=axis, keepdims=keepdims) , cls(t1, axis=axis)

    def __constant(self) -> float:
        p = self.parents[0]
        if self.axis is None: 
            return 1 / math.prod(p.shape)
        elif isinstance(self.axis, int): 
            return 1 / p.shape[self.axis]
        return 1 / math.prod(p.shape[i] for i in self.axis)

    def backward(self, partial: NDArray):
        p = self.parents[0]
        prob = self.__constant()
        if p.requires_grad:
            p.grad += expand(partial, p.shape, self.axis) * np.full_like(p.data, prob)

    def __str__(self):
        axis = "()" if self.axis is None else f"(axis = {self.axis})"
        return f'Mean{axis}' if not self._name else self._name    