from giagrad.tensor import Tensor
from giagrad.nn.module import Module

class Linear(Module):
	def __init__(self, in_features: int, out_features: int, bias: bool = True):
		super().__init__()
		self.w = Tensor.uniform(out_features, in_features, requires_grad=True)
		if bias:
			self.b = Tensor.uniform(out_features, 1, requires_grad=True)

	def __call__(self, x) -> Tensor:
		# Assumes x are column vectors
		return self.w @ x + self.b

