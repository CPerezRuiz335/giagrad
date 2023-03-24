from giagrad.tensor import Tensor
from giagrad.nn.module import Module
from math import sqrt

class Linear(Module):
	def __init__(self, in_features: int, out_features: int, bias: bool = True):
		super().__init__()
		stdev = 1 / sqrt(in_features)
		self.w = Tensor.empty(out_features, in_features, requires_grad=True).uniform(a=-stdev, b=stdev)
		if bias:
			self.b = Tensor.empty(1, out_features, requires_grad=True).uniform(a=-stdev, b=stdev)

	def __call__(self, x) -> Tensor:
		# x are row vectors
		return x @ self.w.T + self.b

	def __str__(self):
		out, in_ = self.w.shape
		return f"Layer(in = {in_}, out = {out})"

