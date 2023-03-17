from giagrad.tensor import Tensor
from giagrad.nn.module import Module

class Linear(Module):
	def __init__(self, in_features: int, out_features: int, bias: bool = True):
		super().__init__()
		self.w = Tensor.uniform(out_features, in_features, requires_grad=True)
		if bias:
			self.b = Tensor.uniform(1, out_features, requires_grad=True)

	def __call__(self, x) -> Tensor:
		# x are row vectors
		return x @ self.w.T + self.b

	def __str__(self):
		out, in_ = self.w.shape
		return f"Layer(in = {in_}, out = {out})"

