from typing import List
from collections import OrderedDict
from numpy.typing import NDArray
from giagrad.tensor import Tensor

class Module:
	def __new__(cls, *args, **kwargs):
		instance = super().__init__(cls)
		instance.__odict__ = OrderedDict()
		return instance 

	def __setattr__(self, key, value):
		self.__odict__[key] = value
		super().__setattr__(self, key, value)

	def parameters(self) -> List[Tensor]:
		fn = lambda x: x.parameters() if isinstance(x, Module) else x
		return sum([fn(p) for p in self.__odict__.values()], [])

	def zero_grad(self):
		for t in self.parameters():
			t.grad = np.zeros_like(t.grad)