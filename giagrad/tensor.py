from __future__ import annotations
import numpy as np 
from typing import List, Tuple, Callable, Optional, Literal, Type, Union, Set
import giagrad.operations as ops

class Tensor:

	"""
	Attributes
	----------
	data: np.ndarray
		weights
	grad: NDArray
		gradients
	requires_grad: bool
		indicates if automatic differentiation is needed
	name: Optional[str]
		variable name, for visualization
	_ctx: Context
		class Context: defines parent nodes and function that created it
			like _prev and _op in micrograd.
	"""

	__slots__ = ["data", "grad", "_ctx", "requires_grad", "name"]

	def __init__(self, data, requires_grad: bool = False, context: Context = None):
		super().__init__()
		self.data = np.array(data)
		self.grad = None
		self._ctx = context
		self.requires_grad = requires_grad
		self.name: Optional[str] = ''
    
 
    
	# ****** backprop ******
	def backward(self):
		assert self.requires_grad, "can't backprop when grad is not required"
		"""https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
		a.k.a topological sort / postorder then reversed
		"""
		topo = []
		visited = set()
		
		def build_topo(tensor: Tensor):
			if (context := tensor._ctx):
				for t in context.parents:
					if t not in visited:
						build_topo(t)

				topo.append(tensor)

		build_topo(self)

		# chain rule 
		self.grad = Tensor.ones_like(self) # dL/dL = 1
		for tensor in reversed(topo):
			tensor._ctx.backward(tensor.grad)
			del tensor._ctx # free memory

			

	# ****** idk methods *******
	def shape(self) -> Tuple[int, ...]: return self.data.shape
	def dtype(self) -> type: return np.float32
	def no_grad(self): self.requires_grad = False
	def requires_grad_(self): self.requires_grad = True

	def __repr__(self):
		return str(self.data)

	# ****** creation helpers *******
	# https://github.com/geohot/tinygrad/blob/master/tinygrad/tensor.py

	@classmethod
	def zeros_like(cls, tensor, **kwargs): 
		return cls.zeros(*tensor.shape, **kwargs)

	@classmethod
	def zeros(cls, *shape, **kwargs): 
		return cls(np.zeros(shape, dtype=np.float32), **kwargs)
	
	@classmethod
	def ones_like(cls, tensor, **kwargs):
		return cls(np.ones(*tensor.shape, dtype=np.float32), **kwargs)

	@classmethod
	def ones(cls, *shape, **kwargs): 
		return cls(np.ones(shape, dtype=np.float32), **kwargs)
	
	@classmethod
	def empty(cls, *shape, **kwargs): 
		return cls(np.empty(shape, dtype=np.float32), **kwargs)
	
	@classmethod
	def randn(cls, *shape, **kwargs): 
		return cls(np.random.default_rng().standard_normal(size=shape, dtype=np.float32), **kwargs)
	
	@classmethod
	def arange(cls, stop, start=0, **kwargs): 
		return cls(np.arange(start=start, stop=stop, dtype=np.float32), **kwargs)

	@classmethod
	def uniform(cls, *shape, **kwargs): 
		return cls((np.random.default_rng().random(size=shape, dtype=np.float32) * 2 - 1), **kwargs)

	@classmethod
	def scaled_uniform(cls, *shape, **kwargs): 
		return cls((np.random.default_rng().random(size=shape, dtype=np.float32) * 2 - 1) \
			* (prod(shape)**-0.5), **kwargs)

	@classmethod
	def glorot_uniform(cls, *shape, **kwargs): 
		return cls((np.random.default_rng().random(size=shape, dtype=np.float32) * 2 - 1) \
			* ((6/(shape[0]+np.prod(shape[1:])))**0.5), **kwargs)

	@classmethod
	def eye(cls, dim, **kwargs): 
		return cls(np.eye(dim, dtype=np.float32), **kwargs)

	import giagrad.operations as ops
    
	# ****** math functions (unary) ****** 
	def __neg__(self): return Tensor(-self.data, self.requires_grad, self._ctx)
	def sqrt(self): return self.pow.forward(0.5)
	def square(self): return self*self
	# TODO
	def clip(self, min_, max_): return ((self-min_).relu()+min_) - (self-max_).relu()
	def abs(self): return self.relu() + (-self).relu()
	def sign(self): return self / (self.abs() + 1e-10)

	# ***** activation functions (unary) ***** 
	# TODO
	def sigmoid(self): return (1.0 + (-self).exp()).reciprocal()
	def elu(self, alpha=1.0): return self.relu() - alpha*(1-self.exp()).relu()
	def swish(self): return self * self.sigmoid()
	def silu(self): return self.swish()   # The SiLU function is also known as the swish function.
	def relu6(self): return self.relu() - (self-6).relu()
	def hardswish(self): return self * (self+3).relu6() * (1/6)
	def tanh(self): return 2.0 * ((2.0 * self).sigmoid()) - 1.0
	def gelu(self): return 0.5 * self * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())
	def quick_gelu(self): return self * (self * 1.702).sigmoid()
	def leakyrelu(self, neg_slope=0.01): return self.relu() - (-neg_slope*self).relu()
	def mish(self): return self * self.softplus().tanh()
	def softplus(self, limit=20, beta=1): return (1/beta) * (1 + (self*beta).exp()).log()

	# ***** first class ops (mlops) *****
	# TODO
	def relu(self): return mlops.ReLU.apply(self)
	def log(self): return mlops.Log.apply(self)
	def exp(self): return mlops.Exp.apply(self)
	def reciprocal(self): return mlops.Reciprocal.apply(self)

	# ***** math functions (binary) ******
	def __add__(self, x): return ops.Add.forward(self, x)
	def __radd__(self, x): return ops.Add.forward(x, self)
	def __sub__(self, x): return ops.Sub.forward(self, x)
	def __rsub__(self, x): return ops.Sub.forward(x, self)
	def __mul__(self, x): return ops.Mul.forward(self, x)
	def __rmul__(self, x): return ops.Mul.forward(x, self)
	def __pow__(self, x): return ops.Pow.forward(self, x)
	def __rpow__(self, x): return ops.Pow.forward(x, self)
	def __matmul__(self, x): return ops.Matmul.forward(self, x)
	def __rmatmul__(self, x): return ops.Matmul.forward(x, self)
	# TODO
	def __truediv__(self, x): return self * (x.reciprocal() if isinstance(x, Tensor) else (1/x))
	def __rtruediv__(self, x): return self.reciprocal() * x

	# ***** math functions autossign (i.e. a += b) *******
	# hint: check if leaf node (creates cycles, avoid)
	# TODO
	def __iadd__(self, x): return self.assign(self.__add__(x))
	def __isub__(self, x): return self.assign(self.__sub__(x))
	def __imul__(self, x): return self.assign(self.__mul__(x))
	def __ipow__(self, x): return self.assign(self.__pow__(x))
	def __itruediv__(self, x): return self.assign(self.__truediv__(x))
	def __imatmul__(self, x): self.assign(self.__matmul__(x))

	# simple tensor math API
	def add(self, x): return self.__add__(x)
	def sub(self, x): return self.__sub__(x)
	def mul(self, x): return self.__mul__(x)
	def pow(self, x): return self.__pow__(x)
	def matmul(self, x): return self.__matmul__(x)
	# TODO
	def div(self, x): return self.__truediv__(x)