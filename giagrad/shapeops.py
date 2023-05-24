from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple, TypeVar, Sequence, Union, List, Optional
from giagrad.tensor import Function

"""
Numpy tranpose creates a view, but we are 
interested in changing grad. From an operator
we can not do that. 
"""

class _Permute(Function):
    def __init__(self, axes: Tuple[int, ...]):
        super().__init__()
        self.axes = axes
        self._name += f"(axes = {self.axes})"

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        self.axes = tuple(range(t1.ndim))[::-1] if self.axes is None else self.axes
        return np.transpose(t1.data, axes=self.axes)

    def backward(self, partial: NDArray):
        """Partial is already p.grad but not unpermuted"""
        p = self.parents[0]
        if p.requires_grad:
            p.grad += np.transpose(partial, np.argsort(self.axes))

class _Getitem(Function):
    def __init__(self, idx: Union[Sequence, Tuple[int, ...], List[int], int]):
        super().__init__()
        self.idx = idx

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        return t1.data[self.idx]

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad[self.idx] += partial

class _Pad(Function):
    def __init__(
        self, 
        # padding is either a single int or a tuple, that tuple
        # can be made of tuples of pairs of ints or single ints 
        padding: Union[Tuple[Union[Tuple[int, int], int], ...], int],
        mode: str,
        **kwargs
    ):
        super().__init__()
        # format padding so that numpy.pad accepts it
        if isinstance(padding, int):
            self.padding = ((padding,)*2,)
        elif isinstance(padding, tuple):
            self.padding = tuple(i if isinstance(i, tuple) else (i, i) for i in padding)

        for arg in ['stat_length', 'constant_values', 'end_values']:
            if arg in kwargs and isinstance(kwargs[arg], int):
                kwargs[arg] = ((kwargs[arg],)*2)
            elif arg in kwargs:
                kwargs[arg] = tuple(i if isinstance(i, tuple) else (i, i) for i in kwargs[arg])
        self.kwargs = kwargs
        self.mode = mode 
        self._name = mode.capitalize() + self._name

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        return np.pad(
            t1.data, 
            pad_width=((0,0),)*(t1.ndim - len(self.padding)) + self.padding, 
            mode=self.mode,
            **self.kwargs
        )

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            idx = (...,) + tuple(
                slice(start, -end) if start > 0 and end > 0 else
                slice(-end) if start == 0 else
                slice(start, None) if end == 0 else slice(None)
                for start, end in self.padding 
            )
            p.grad += partial[idx]

class _Swapaxes(Function):
    def __init__(self, axis0: int, axis1: int):
        super().__init__()
        self.axis0 = axis0
        self.axis1 = axis1
        self._name += f"({self.axis0}, {self.axis1})"

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        return np.swapaxes(t1.data, self.axis0, self.axis1)

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad += np.swapaxes(partial, self.axis1, self.axis0)

class _Squeeze(Function):
    def __init__(self, axis: Optional[Union[Tuple[int, ...], int]]):
        super().__init__()
        self.axis = axis
        self._name += f"(axis = {self.axis})"

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        if self.axis is None:
            self.axis = tuple(i for i, ax in enumerate(t1.shape) if ax == 1)
        return np.squeeze(t1.data, self.axis)

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad += np.expand_dims(partial, self.axis)

class _UnSqueeze(Function):
    def __init__(self, axis: Union[Tuple[int, ...], int]):
        super().__init__()
        self.axis = axis
        self._name += f"(axis = {self.axis})"

    def forward(self, t1) -> NDArray:
        self.save_for_backward(t1)
        return np.expand_dims(t1.data, self.axis)

    def backward(self, partial: NDArray):
        p = self.parents[0]
        if p.requires_grad:
            p.grad += np.squeeze(partial, self.axis)
