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
		p1 = self.parents[0]
		if p1.requires_grad:
			p1.grad += partial * np.maximum(p1.data, 0).astype(bool)

	def __str__(self):
		return f"ReLU"
"""
TODO
----
Activation functions
Max pooling
...
"""