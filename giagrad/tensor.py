from __future__ import annotations
import numpy as np 
from numpy.typing import NDArray
from typing import List, Tuple, Callable, Optional, Literal, Type, Union, Set, Any
from abc import ABC, abstractmethod

class Context(ABC):
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
    def __init__(self, save_for_backward: Tuple[Tensor, ...]):
        self.parents = save_for_backward
        self._name = None
        super().__init__()

    @classmethod
    @abstractmethod
    def forward(cls, *tensors, **kwargs) -> Tuple[Union[NDArray, float], Context]:
        """Main function for forward pass."""
        raise NotImplementedError(f"forward not implemented for {type(cls)}")
    
    @abstractmethod
    def backward(self, partial: NDArray):
        """Backprop automatic differentiation, to update grad of parents.
        partial: gradient of the output of forward method."""
        raise NotImplementedError(f"backward not implemented for {type(self)}")

    @abstractmethod
    def __str__(self):
        """For graphviz visualization."""
        raise NotImplementedError(f"__str__ not implemented for class {type(self)}")

import giagrad.shapeops as sops
import giagrad.mathops as mops
import giagrad.reductionops as rops
import giagrad.mlops as mlops
import giagrad.initializers as init

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

    def __init__(
            self, 
            data, 
            requires_grad: bool = False, 
            context: Optional[Context] = None, 
            name: str = '',
            dtype = np.float32):
        self.data = np.array(data, dtype=dtype)
        self.grad = np.zeros_like(self.data)
        self._ctx = context
        self.requires_grad = requires_grad
        self.name = name
    
    # ***** backprop *****
    def backward(self, debug: bool = False):
        """https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
        a.k.a topological sort / postorder then reversed
        """
        topo = []
        visited = set()
        
        def build_topo(tensor: Tensor):
            if (context := tensor._ctx):
                for t in context.parents:
                    if t not in visited:
                        visited.add(t)
                        build_topo(t)
                topo.append(tensor)

        build_topo(self)
        # chain rule 
        self.grad = np.ones(self.shape) # dL/dL = 1

        for tensor in reversed(topo):
            tensor._ctx.backward(tensor.grad)
            if not debug: del tensor._ctx # free memory

    # ***** helpers *****
    @property
    def shape(self) -> Tuple[int, ...]: return self.data.shape
    
    @property
    def dtype(self) -> type: return self.data.dtype

    @property
    def size(self) -> int: return self.data.size

    @property
    def dim(self) -> int: return self.data.ndim

    def no_grad(self) -> Tensor: 
        self.requires_grad = False
        return self

    def requires_grad_(self) -> Tensor: 
        self.requires_grad = True
        return self

    def __repr__(self):
        return f"gtensor(\ndata: {np.array_str(self.data, precision=4)}," \
                + f"\ngrad: {np.array_str(self.grad, precision=4)},"\
                + f"\ngrad_fn: {self._ctx})"

    def __str__(self):
        return f"gtensor({np.array_str(self.data, precision=4)}," \
                + f"grad_fn: {self._ctx})"

    # ***** initializers in-place*****
    # use empty as creator and modify it by in-place methods

    @classmethod
    def empty(cls, *shape, **kwargs) -> Tensor: return cls(np.empty(shape), **kwargs)

    # in-place initializers
    def zeros(self): self.data = np.zeros_like(self.data)
    def ones(self): self.data = np.ones_like(self.data)
    def constant(self, val): self.data = np.full_like(self.data, fill_value=val)

    def normal(self, mu, sigma): init.normal(self, mu, sigma)
    def uniform(self, a, b): init.uniform(a, b)
    def dirac(self, groups=1): init.dirac(self, groups=groups)
    def xavier_uniform(self, gain=1): init.xavier_uniform(self, gain=gain)
    def xavier_normal(self, gain=1): init.xavier_normal(self, gain=gain)
    
    def kaiming_uniform(self, a=0, mode='fan_in', nonlinearity='leaky_relu'): 
        nit.kaiming_uniform(self, a, mode, nonlinearity)
    def kaiming_normal(self, a=0, mode='fan_in', nonlinearity='leaky_relu'): 
        init.kaiming_normal(self, a, mode, nonlinearity)
    
    def sparse(self, sparsity, std=0.01): init.sparse(self, sparsity, std)
    def orthogonal(self, gain=1): init.orthogonal(self, gain)

    ### MATH ###
    @classmethod
    def comm(cls, operator: Context, *tensors, **kwargs) -> Tensor:
        # Avoids circular imports between tensor.py and operations.py
        operands = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
        data, context = operator.forward(*operands, **kwargs)
        return cls(data, requires_grad=True, context=context)
    

    # ***** math functions (unary) ***** 
    def sqrt(self): return self.pow(0.5)
    def square(self): return self.pow(2)
    def exp(self): return Tensor.comm(mops.Exp, self)
    def log(self): return Tensor.comm(mops.Log, self)
    def reciprocal(self): return Tensor.comm(mops.Reciprocal, self)
    def abs(self): return Tensor.comm(mops.Abs, self) 
    def __neg__(self): return 0.0-self # Tensor.comm(mops.Mul, self, -1)
    # TODO
    def clip(self, min_, max_): raise NotImplementedError() # ((self-min_).relu()+min_) - (self-max_).relu()
    def sign(self): raise NotImplementedError() # return self / (self.abs() + 1e-10)

    # ***** activation functions (unary) ***** 
    def relu(self): return Tensor.comm(mlops.ReLU, self) 
    def sigmoid(self): return Tensor.comm(mlops.Sigmoid, self) 
    def elu(self, alpha=1.0): return Tensor.comm(mlops.ELU, self, alpha=alpha) 
    def silu(self, beta=1.0): return Tensor.comm(mlops.SiLU, self, beta=beta)
    def tanh(self): return Tensor.comm(mlops.Tanh, self)
    def leakyrelu(self, neg_slope=0.01): return Tensor.comm(mlops.LeakyReLU, self, neg_slope=neg_slope)
    def softplus(self, limit=20, beta=1): return Tensor.comm(mlops.Softplus, self, limit=limit, beta=beta)
    def quick_gelu(self): return Tensor.comm(mlops.SiLU, self, beta=1.702)
    def gelu(self): return Tensor.comm(mlops.GELU, self) 
    def relu6(self): return Tensor.comm(mlops.ReLU6, self) 
    def mish(self): return self * self.softplus().tanh()
    def hardswish(self): return Tensor.comm(mlops.Hardswish, self)
    def softmax(self, axis: int): return Tensor.comm(mlops.Softmax, self, axis=axis)
    def log_softmax(self, axis: int): return Tensor.comm(mlops.LogSoftmax, self, axis=axis)

    # ***** math functions (binary) *****
    def __add__(self, x): return Tensor.comm(mops.Add, self, x)
    def __radd__(self, x): return Tensor.comm(mops.Add, x, self)
    def __sub__(self, x): return Tensor.comm(mops.Sub, self, x)
    def __rsub__(self, x): return Tensor.comm(mops.Sub, x, self)
    def __mul__(self, x): return Tensor.comm(mops.Mul, self, x)
    def __rmul__(self, x): return Tensor.comm(mops.Mul, x, self)
    def __pow__(self, x): return Tensor.comm(mops.Pow, self, x)
    def __rpow__(self, x): return Tensor.comm(mops.Pow, x, self)
    def __matmul__(self, x): return Tensor.comm(mops.Matmul, self, x)
    def __rmatmul__(self, x): return Tensor.comm(mops.Matmul, x, self)
    def __truediv__(self, x): return Tensor.comm(mops.Div, self, x) 
    def __rtruediv__(self, x): return Tensor.comm(mops.Div, x, self) 

    # ***** math functions autossign (i.e. a += b) *****
    def __iadd__(self, x): self.data += x.data if isinstance(x, Tensor) else x; return self
    def __isub__(self, x): self.data -= x.data if isinstance(x, Tensor) else x; return self
    def __imul__(self, x): self.data *= x.data if isinstance(x, Tensor) else x; return self
    def __ipow__(self, x): self.data **= x.data if isinstance(x, Tensor) else x; return self
    def __itruediv__(self, x): self.data /= x.data if isinstance(x, Tensor) else x; return self
    def __imatmul__(self, x): self.data = self.data @ x.data if isinstance(x, Tensor) else x; return self

    # ***** math functions (reduction) *****
    def mean(self, dim: Union[Tuple[int, ...], int, None] = None, keepdim: bool = False): 
        return Tensor.comm(rops.Mean, self, axis=dim, keepdims=keepdim)
    
    def sum(self, dim: Union[Tuple[int, ...], int, None] = None, keepdim: bool = False): 
        return Tensor.comm(rops.Sum, self, axis=dim, keepdims=keepdim)
    
    def max(self, dim: Union[Tuple[int, ...], int, None] = None, keepdim: bool = False): 
        return Tensor.comm(rops.MinMax, self, axis=dim, keepdims=keepdim, fn=np.max)

    def min(self, dim: Union[Tuple[int, ...], int, None] = None, keepdim: bool = False): 
        return Tensor.comm(rops.MinMax, self, axis=dim, keepdims=keepdim, fn=np.min)
    # simple tensor math API
    def add(self, x): return self.__add__(x)
    def sub(self, x): return self.__sub__(x)
    def mul(self, x): return self.__mul__(x)
    def pow(self, x): return self.__pow__(x)
    def matmul(self, x): return self.__matmul__(x)
    def div(self, x): return self.__truediv__(x)

    # ***** shape functions (reduction) *****
    # this operators create views
    def permute(self, axes=None): return self.comm(sops.Permute, self, axes=axes)
    def transpose(self, dim0, dim1): return self.comm(sops.Permute, self, axes=(dim1, dim0))
    @property
    def T(self): 
        assert self.dim == 2, "Dimensions = 2 required, this is matrix transposition" 
        return self.comm(sops.Permute, self, axes=(1, 0))

