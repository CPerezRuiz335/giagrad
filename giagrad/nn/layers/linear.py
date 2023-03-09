from giagrad.nn.module import Module
from giagrad.tensor import Tensor

class Linear(Module):
	def __init__(self, in_features: int, out_features: int, bias: bool = True):
		super().__init__()
		self.w = Tensor.uniform((in_features, out_features))
		if bias:
			self.b = Tensor.uniform((out_features, 1))

	def __call__(self, x) -> Tensor:
		return self.w @ x + self.b

