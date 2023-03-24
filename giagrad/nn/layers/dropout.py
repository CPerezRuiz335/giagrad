from giagrad.tensor import Tensor
from giagrad.nn.module import Module
import numpy as np


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

class Dropout1d(Module):
	def __init__(self, p: float = 0.5):
		super().__init__()
		self.p = p
	
	def __call__(self, x: Tensor) -> Tensor:
		if x.ndim not in [2,3]:
			raise ValueError("Dropout 1d only for 2D and 3D tensors")
		if x.ndim == 2:
			pass
		else:
			pass

class Dropout2d(Module):
	def __init__(self, p: float):
		super().__init__()
		self.p = p
	
class Dropout3d(Module):
	def __init__(self, p: float):
		super().__init__()
		self.p = p
