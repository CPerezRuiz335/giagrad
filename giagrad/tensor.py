from __future__ import annotations
import numpy as np 
from numpy.typing import NDArray
from typing import List, Tuple, Callable, Optional, Literal, Type, Union, Set, Any

class Context:
    """
    Abstract class for all operators defined in mathops, reductionsops, etc
    An operator creates an instance of itself with class method forward that
    contains the parents of the Tensor created by Tensor.comm()

    Operators are just extensions of Tensor class to have Tensor functionality 
    self contained but separated in different files. 

    Attributes
    ----------
    parents: Tuple[Any, ...]
        operands/Tensors of the operator, can contain other values with Tensor.comm(.., **kwargs)

    """
    def __init__(self, save_for_backward: Tuple[Any, ...]):
        self.parents = save_for_backward

    @classmethod
    def forward(cls, *tensors, **kwargs) -> Tuple[Union[NDArray, float], Context]:
        """Main function for forward pass."""
        raise NotImplementedError(f"forward not implemented for {type(cls)}")
    
    def backward(self, grad_output: NDArray):
        """Backprop automatic differentiation, to update grad of parents.
        grad_output: gradient of the output of forward method."""
        raise NotImplementedError(f"backward not implemented for {type(self)}")

    def __str__(self):
        """For graphviz visualization."""
        raise NotImplementedError(f"__str__ not implemented for class {type(self)}")

import giagrad.mathops as mops
import giagrad.reductionops as rops


class Tensor:
    # tell numpy to trust Tensor to make __r***__ method
    __array_ufunc__ = None 

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

    def __init__(self, data, requires_grad: bool = False, context: Optional[Context] = None):
        super().__init__()
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._ctx = context
        self.requires_grad = requires_grad
        self.name: Optional[str] = ''
    
    # ****** backprop ******
    def backward(self):
        """https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
        a.k.a topological sort / postorder then reversed
        """
        topo = []
        visited = set()
        
        def build_topo(tensor: Tensor):
            if (context := tensor._ctx):
                for t in context.parents:
                    # _ctx may save other unhashable types of data
                    if isinstance(t, Tensor) and t not in visited:
                        build_topo(t)

                topo.append(tensor)

        build_topo(self)
        # chain rule 
        self.grad = np.ones(self.shape) # dL/dL = 1
        for tensor in reversed(topo):
            if isinstance(tensor._ctx, rops.Reduction): # see reductionops.py
                tensor.grad += np.array(tensor.grad.sum(), dtype=np.float32)
            tensor._ctx.backward(tensor.grad)
            del tensor._ctx # free memory

    # ****** helpers *******
    @property
    def shape(self) -> Tuple[int, ...]: return self.data.shape
    
    @property
    def dtype(self) -> type: return np.float32

    def no_grad(self) -> Tensor: 
        self.requires_grad = False
        return self

    def requires_grad_(self) -> Tensor: 
        self.requires_grad = True
        return self

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
        return cls(np.ones(tensor.shape, dtype=np.float32), **kwargs)

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

    ### MATH ###
    @classmethod
    def comm(cls, operator: Context, *tensors, **kwargs) -> Tensor:
        # Avoids circular imports between tensor.py and operations.py
        operands = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
        data, context = operator.forward(*operands, **kwargs)
        return cls(data, requires_grad=True, context=context)
    
    # ****** math functions (unary) ****** 
    def sqrt(self): return self.pow(0.5)
    def square(self): return self.pow(2)
    def exp(self): return Tensor.comm(mops.Exp, self)
    def mean(self): return Tensor.comm(rops.Mean, self)
    def log(self): return Tensor.comm(mops.Log, self)
    def reciprocal(self): return Tensor.comm(mops.Reciprocal, self)
    def abs(self): raise Tensor.comm(mops.Abs, self) 
    # TODO
    def clip(self, min_, max_): raise NotImplementedError() # ((self-min_).relu()+min_) - (self-max_).relu()
    def sign(self): raise NotImplementedError() # return self / (self.abs() + 1e-10)

    # ***** activation functions (unary) ***** 
    # TODO
    def relu(self): raise NotImplementedError() #  mlops.ReLU.apply(self)
    def sigmoid(self): raise NotImplementedError() #  (1.0 + (-self).exp()).reciprocal()
    def elu(self, alpha=1.0): raise NotImplementedError() # self.relu() - alpha*(1-self.exp()).relu()
    def swish(self): raise NotImplementedError() # self * self.sigmoid()
    def silu(self): raise NotImplementedError() # self.swish()   # The SiLU function is also known as the swish function.
    def relu6(self): raise NotImplementedError() # self.relu() - (self-6).relu()
    def hardswish(self): raise NotImplementedError() # self * (self+3).relu6() * (1/6)
    def tanh(self): raise NotImplementedError() # 2.0 * ((2.0 * self).sigmoid()) - 1.0
    def gelu(self): raise NotImplementedError() # 0.5 * self * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())
    def quick_gelu(self): raise NotImplementedError() # self * (self * 1.702).sigmoid()
    def leakyrelu(self, neg_slope=0.01): raise NotImplementedError() # self.relu() - (-neg_slope*self).relu()
    def mish(self): raise NotImplementedError() # self * self.softplus().tanh()
    def softplus(self, limit=20, beta=1): raise NotImplementedError() # (1/beta) * (1 + (self*beta).exp()).log()

    # ***** math functions (binary) ******
    def __add__(self, x): return Tensor.comm(mops.Add, self, x)
    def __radd__(self, x): return Tensor.comm(mops.Add, x, self)
    def __sub__(self, x): return Tensor.comm(mops.Sub, self, x)
    def __rsub__(self, x): return Tensor.comm(mops.Sub, x, self)
    def __mul__(self, x): return Tensor.comm(mops.Mul, self, x)
    def __rmul__(self, x): return Tensor.comm(mops.Mul, x, self)
    def __pow__(self, x): return Tensor.comm(mops.Pow, self, x)
    def __rpow__(self, x): return Tensor.comm(mops.Pow, x, self)
    def __matmul__(self, x): return Tensor.comm(mops.Matmul, self, x)
    def __rmatmul__(self, x): return Tensor.comm(mops.Matmul, self, x)
    def __neg__(self): return 0.0-self # Tensor.comm(mops.Mul, self, -1)
    # TODO
    def __truediv__(self, x): raise NotImplementedError() # self * (x.reciprocal() if isinstance(x, Tensor) else (1/x))
    def __rtruediv__(self, x): raise NotImplementedError() # self.reciprocal() * x

    # ***** math functions autossign (i.e. a += b) *******
    # hint: check if leaf node (creates cycles, avoid)
    # TODO
    def __iadd__(self, x): raise NotImplementedError() # self.assign(self.__add__(x))
    def __isub__(self, x): raise NotImplementedError() # self.assign(self.__sub__(x))
    def __imul__(self, x): raise NotImplementedError() # self.assign(self.__mul__(x))
    def __ipow__(self, x): raise NotImplementedError() # self.assign(self.__pow__(x))
    def __itruediv__(self, x): raise NotImplementedError() # self.assign(self.__truediv__(x))
    def __imatmul__(self, x): raise NotImplementedError() #self.assign(self.__matmul__(x))

    # simple tensor math API
    def add(self, x): return self.__add__(x)
    def sub(self, x): return self.__sub__(x)
    def mul(self, x): return self.__mul__(x)
    def pow(self, x): return self.__pow__(x)
    def matmul(self, x): return self.__matmul__(x)
    def sum(self): return Tensor.comm(rops.Sum, self)
    def max(self): return Tensor.comm(rops.Max, self)
    def min(self): return Tensor.comm(rops.Min, self)
    # TODO
    def div(self, x): raise NotImplementedError() # self.__truediv__(x)


