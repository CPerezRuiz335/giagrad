from typing import List
from collections import OrderedDict
import numpy as np
from numpy.typing import NDArray
from giagrad.tensor import Tensor
from abc import ABC, abstractmethod

class Module(ABC):
	def __new__(cls, *args, **kwargs):
		instance = object.__new__(cls)
		instance.__odict__ = OrderedDict()
		return instance 

	def __setattr__(self, key, value):
		if key != '__odict__':
			self.__odict__[key] = value
		object.__setattr__(self, key, value)

	def parameters(self) -> List[Tensor]:
		fn = lambda x: x.parameters() if isinstance(x, Module) else x
		return [fn(p) for p in self.__odict__.values()]

	def zero_grad(self):
		for t in self.parameters():
			t.grad = np.zeros_like(t.grad)

	@abstractmethod
	def __call__(self, x) -> Tensor:
		raise NotImplementedError(f"__call__ not implemented in {type(self)}")