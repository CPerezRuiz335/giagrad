from giagrad.tensor import Tensor
from giagrad.nn.module import Module
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import math

def _random_dims_to_zero(r: NDArray, p: float):
	n = math.prod(r.shape[:2]) if r.ndim > 2 else r.shape[0]
	rng = np.random.default_rng()

	for _ in range(n):
		if rng.random() > p:
			if r.ndim == 2:
				ij = rng.integers(0, r.shape[0]) 
			else:
				ij = rng.integers((0, 0), r.shape[:2]) 
			r[tuple(ij)] *= 0


class Dropout(Module):
	def __init__(self, p: float = 0.5):
		super().__init__()
		self.p = p
	
	def __call__(self, x: Tensor) -> Tensor:
		r = np.random.binomial(1, self.p, size=x.shape)
		if self._train:
			return x * r * (1 / (1 - self.p))
		return x			

	def __str__(self):
		return f"Dropout(p = {self.p})"

class DropoutNd(Module):
	def __init__(self):
		super().__init__()
		self.p: float

	def __check(self, ndim: int):
		pass

	def __call__(self, x: Tensor) -> Tensor:
		self.__check(x.ndim)

		if self._train:
			r = np.ones_like(x.data)
			_random_dims_to_zero(r, self.p)
			return x * r * (1 / (1 - self.p))
		return x

class Dropout1d(DropoutNd):
	def __init__(self, p: float = 0.5):
		super().__init__()
		self.p = p
	
	def __check(self, ndim: int):
		if ndim not in [2,3]:
			raise ValueError("Dropout 1d only for 2D and 3D tensors")

class Dropout2d(DropoutNd):
	def __init__(self, p: float):
		super().__init__()
		self.p = p

	def __check(self, ndim: int):
		if ndim not in [3,4]:
			raise ValueError("Dropout 2d only for 3D and 4D tensors")
	
class Dropout3d(DropoutNd):
	def __init__(self, p: float):
		super().__init__()
		self.p = p

	def __check(self, ndim: int):
		if ndim not in [4,5]:
			raise ValueError("Dropout 3d only for 4D and 5D tensors")
