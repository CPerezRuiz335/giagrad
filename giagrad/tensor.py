from __future__ import annotations
import numpy as np 
from numpy.typing import NDArray
from typing import List, Tuple, Callable, Optional, Literal, Type, Union, Set, Any
from abc import ABC, abstractmethod

class Context(ABC):
    """
    Abstract class for all Tensor operators.
    
    Operators extend the Tensor class to provide additional 
    functionality. The Context behavior is accessed through the 
    :func:`~giagrad.Tensor.comm` [1]_ method. To mantain modularity,
    the operators are implemented in separate files.

    Attributes
    ----------
    parents: Tuple[Tensor, ...]
        Tensor or Tensors needed for the child class that inherits Context. 
        :attr:`~parents` should not contain other types than Tensor, if 
        other attributes are needed they should be an instance variable, e.g :math:`a`
        variable for Leaky ReLU

    _name: Optional[str]
        Useful for complex modules that use multiple methods from Tensor class
        and want to override the name of the last operator that created an instance of
        Tensor. Particularly useful for improving the readability of Tensor
        through __repr__ method
    """
    def __init__(self, save_for_backward: Tuple[Tensor, ...]):
        self.parents = save_for_backward
        self._name = None
        super().__init__()

    @classmethod
    @abstractmethod
    def forward(cls, *tensors, **kwargs) -> Tuple[Union[NDArray, float], Context]:
        """
        Makes forward pass.

        Parameters
        ----------
        *tensors: Tensor
            input tensors, e.g. two for binary operators such as :func:`~giagrad.Tensor.matmul`

        *kwargs: Any
            optional arguments if needed
        """
        raise NotImplementedError(f"forward not implemented for {type(cls)}")
    
    @abstractmethod
    def backward(self, partial: NDArray):
        """
        Backpropagate from child Tensor node created with :func:`~giagrad.Tensor.comm`.
        
        Updates parents' gradient through chain rule. This method is the extension
        of :func:`~giagrad.Tensor.backward` for a concrete operator.

        Parameters
        ----------
        partial: ndarray
            defines the partial derivative of the loss function with respect to the 
            Tensor derived from parents
        """
        raise NotImplementedError(f"backward not implemented for {type(self)}")

    @abstractmethod
    def __str__(self):
        """
        Default representation if :attr:`~_name` is not overriden.
        """
        raise NotImplementedError(f"__str__ not implemented for class {type(self)}")

import giagrad.shapeops as sops
import giagrad.mathops as mops
import giagrad.reductionops as rops
import giagrad.mlops as mlops
import giagrad.initializers as init

class Tensor:
    __array_ufunc__ = None # tell numpy to trust Tensor to make __r***__ method
    __slots__ = ["data", "grad", "_ctx", "requires_grad", "name"]

    def __init__(
            self, 
            data: NDArray, 
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
            if not debug: self._ctx = None 

    # ***** helpers *****
    @property
    def shape(self) -> Tuple[int, ...]: 
        """
        Tuple of Tensor dimensions

        Unlike numpy.ndarray.shape it can not be used to 
        reshape inplace.
        """
        return self.data.shape
    
    @property
    def dtype(self) -> type: 
        """Data-type of the Tensor."""
        return self.data.dtype

    @property
    def size(self) -> int: 
        """Size of the Tensor."""
        return self.data.size

    @property
    def ndim(self) -> int: 
        """Number of the Tensor dimensions."""
        return self.data.ndim

    def no_grad(self) -> Tensor: 
        """Makes Tensor autodifferentiable."""
        self.requires_grad = False
        return self

    def requires_grad_(self) -> Tensor:
        """Makes Tensor not autodifferentiable.""" 
        self.requires_grad = True
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'tensor: ' + np.array2string(
            self.data, 
            prefix='tensor: ') \
            + (f" grad_fn: {self._ctx}" if self._ctx else '') \
            + (f", name: {self.name}" if self.name else '')

    # ***** initializers in-place*****
    # use empty as creator and modify it by in-place methods

    @classmethod
    def empty(cls, *shape, **kwargs) -> Tensor: 
        """
        Creates a Tensor filled with uninitialized data. 
    
        Parameters
        ----------
        shape: Tuple[int, ...]
            a variable number of integers defining the shape of the output Tensor
        \*\*kwargs:
            this parameters will be passed to the Tensor initializer
    
        Examples
        --------
            >>> Tensor.empty(2, 3, requires_grad=True, dtype=np.float64)
            tensor: [[4.67662529e-310 0.00000000e+000 4.67596337e-310]
                     [6.94592882e-310 6.94611561e-310 6.94609055e-310]]    
        """
        return cls(np.empty(shape), **kwargs)

    # in-place initializers
    def zeros(self): 
        """
        Fills Tensor data with zeros. 
    
        Examples
        -------_
            >>> Tensor.empty(2, 3).zeros()                                                                                           
            tensor: [[0. 0. 0.]
                     [0. 0. 0.]]  
            >>> Tensor([1, 3, 4, 5]).zeros()
            tensor: [0., 0., 0., 0.] 
        """
        self.data = np.zeros_like(self.data)
        return self

    def ones(self): 
        """
        Fills Tensor data with ones. 
    
        Examples
        --------
            >>> Tensor.empty(2, 3).ones()                                                                                           
            tensor: [[1. 1. 1.]
                     [1. 1. 1.]]  
            >>> Tensor([1, 3, 4, 5]).ones()
            tensor: [1., 1., 1., 1.] 
        """
        self.data = np.ones_like(self.data)
        return self

    def full(self, fill_value): 
        """
        Fills Tensor data with a constant value. 
    
        Parameters
        ----------
        fill_value: Scalar
            the value to fill the output Tensor with

        Examples
        --------
            >>> Tensor.empty(2, 3).fill(2.71828)                                                                                           
            tensor: [[2.71828 2.71828 2.71828]
                     [2.71828 2.71828 2.71828]]  
        """
        self.data = np.full_like(self.data, fill_value=val)
        return self
        
    def normal(self, mu: float = 0.0, sigma: float = 1.0): 
        r"""Fills Tensor data with values drawn from the normal
        distribution :math:`\mathcal{N}(\text{mu}, \text{sigma}^2)`.

        Parameters
        ----------
            mean: float
                the mean of the normal distribution
            sigma: 
                the standard deviation of the normal distribution

        Examples
        --------
            >>> Tensor.empty(3, 3).normal()
        """
        init.normal(self, mu, sigma)
        return self

    def uniform(self, a: float = 0.0, b: float = 1.0):
        r"""Fills Tensor data with values drawn from the uniform
        distribution :math:`\mathcal{U}(a, b)`.

        Args:
            a: float
                the lower bound of the uniform distribution
            b: float
                the upper bound of the uniform distribution

        Examples:
            >>> Tensor.empty(3, 3).uniform()
        """ 
        init.uniform(self, a, b)
        return self

    def dirac(self, groups: int = 1): 
        r"""Fills the {3, 4, 5}-dimensional Tensor data with the Dirac
        delta function. 

        Preserves the identity of the inputs in *Convolutional*
        layers, where as many input channels are preserved as possible. In case
        of groups > 1, each group of channels preserves identity

        Parameters
        ----------
            groups: int
                number of groups in the conv layer 
        
        Examples
        --------
            >>> Tensor.empty(3, 24, 5, 5).dirac(3)
        """
        init.dirac(self, groups=groups)
        return self

    def xavier_uniform(self, gain: float = 1.0): 
        r"""Fills Tensor data with the also known Glorot uniform initialization.

        This methos is described in `Understanding the difficulty of training deep feedforward
        neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
        distribution. Tensor data will have values sampled from :math:`\mathcal{U}(-a, a)` where

        .. math::
            a = \text{gain} \times \sqrt{\frac{6}{\text{fan_in} + \text{fan_out}}}

        Parameters
        ----------
            gain: float 
                an optional scaling factor

        Examples
        --------
            >>> Tensor.empty(3, 5).xavier_uniform(gain=calculate_gain('relu'))
        """
        init.xavier_uniform(self, gain=gain)
        return self
    
    def xavier_normal(self, gain: float = 1.0): 
        r"""Fills Tensor data with the also known Glorot normal initialization.

        This methos is described in `Understanding the difficulty of training deep feedforward
        neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
        distribution. Tensor data will have values sampled from :math:`\mathcal{N}(0, \sigma^2)` where

        .. math::
            \sigma = \text{gain} \times \sqrt{\frac{2}{\text{fan_in} + \text{fan_out}}}

        Parameters
        ----------
            gain: float
                an optional scaling factor

        Examples
        --------
            >>> Tensor.empty(3, 5).xavier_normal(gain=calculate_gain('relu'))
        """
        init.xavier_normal(self, gain=gain)
        return self    

    def kaiming_uniform(
        self, neg_slope: float = 0.0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'
    ): 
        r"""Fills Tensor data with the also known He uniform initialization.

        Tensor data is filled with values according to the method described 
        in `Delving deep into rectifiers`_ using uniform distribution. The 
        resulting tensor will have values sampled from
        :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

        .. math::
            \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan_mode}}}

        Parameters
        ----------
            neg_slope: float
                the negative slope of the rectifier used after this layer (only
                used with `'leaky_relu'`)
            mode: str
                either `'fan_in'` or `'fan_out'`. Choosing `'fan_in'`
                preserves the magnitude of the variance of the weights in the
                forward pass. Choosing `'fan_out'` preserves the magnitudes in the
                backwards pass
            nonlinearity: str
                the non-linear function method name,
                recommended to use only with `'relu'` or `'leaky_relu'`

        Examples
        --------
            >>> Tensor.empty(3, 5).kaiming_uniform(mode='fan_in', nonlinearity='relu')

        .. _Delving deep into rectifiers: https://arxiv.org/abs/1502.01852
        """
        init.kaiming_uniform(self, neg_slope, mode, nonlinearity)
        return self

    def kaiming_normal(
        self, neg_slope: float = 0.0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'
    ): 
        r"""Fills Tensor data with the also known He normal initialization.

        Tensor data is filled with values according to the method described 
        in `Delving deep into rectifiers`_ using normal distribution. The 
        resulting tensor will have values sampled from
        :math:`\mathcal{N}(0, \sigma^2)` where

        .. math::
            \sigma = \frac{\text{gain}}{\sqrt{\text{fan_mode}}}

        Parameters
        ----------
            neg_slope: float
                the negative slope of the rectifier used after this layer (only
                used with `'leaky_relu'`)
            mode: str
                either `'fan_in'` or `'fan_out'`. Choosing `'fan_in'`
                preserves the magnitude of the variance of the weights in the
                forward pass. Choosing `'fan_out'` preserves the magnitudes in the
                backwards pass
            nonlinearity: str
                the non-linear function method name,
                recommended to use only with `'relu'` or `'leaky_relu'`

        Examples
        --------
            >>> Tensor.empty(3, 5).kaiming_normal(mode='fan_in', nonlinearity='relu')

        .. _Delving deep into rectifiers: https://arxiv.org/abs/1502.01852
        """
        init.kaiming_normal(self, neg_slope, mode, nonlinearity)
        return self    

    def sparse(self, sparsity: float, sigma=0.01): 
        r"""Fills the 2D Tensor data as a sparse matrix.

        Non-zero elements will be drawn from the normal distribution
        :math:`\mathcal{N}(0, \text{sigma})`, as described in `Deep learning via
        Hessian-free optimization` - Martens, J. (2010).

        Parameters
        ----------
            sparsity: [0, 1) 
                the fraction of elements in each column to be set to zero
            std: the standard deviation of the normal distribution used to generate
                the non-zero values

        Examples
        --------
            >>> Tensor.empty(3, 5).sparse(sparsity=0.4, sigma=0.2)

        .. _Deep learning via Hessian-free optimization: https://dl.acm.org/doi/10.5555/3104322.3104416
        """
        init.sparse(self, sparsity, sigma) 
        return self

    def orthogonal(self, gain: float = 1.0):
        r"""Fills Tensor data with a (semi) orthogonal matrix.

        Values are generated according to the method described in 
        Exact solutions to the nonlinear dynamics of learning in deep
        linear neural networks - Saxe, A. et al. (2013). The Tensor must have
        at least 2 dimensions, and for Tensors with more than 2 dimensions the
        trailing dimensions are flattened.

        Parameters
        ----------
            gain: Scalar
                optional scaling factor

        Examples:
            >>> Tensor.empty(3, 5).orthogonal()
        """ 
        init.orthogonal(self, gain)
        return self

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
    def mean(self, axis: Union[Tuple[int, ...], int, None] = None, keepdims: bool = False): 
        return Tensor.comm(rops.Mean, self, axis=axis, keepdims=keepdims)
    
    def sum(self, axis: Union[Tuple[int, ...], int, None] = None, keepdims: bool = False): 
        return Tensor.comm(rops.Sum, self, axis=axis, keepdims=keepdims)
    
    def max(self, axis: Union[Tuple[int, ...], int, None] = None, keepdims: bool = False): 
        return Tensor.comm(rops.MinMax, self, axis=axis, keepdims=keepdims, fn=np.max)

    def min(self, axis: Union[Tuple[int, ...], int, None] = None, keepdims: bool = False): 
        return Tensor.comm(rops.MinMax, self, axis=axis, keepdims=keepdims, fn=np.min)
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
        """Returns a transposed view of a 2 dimensional Tensor."""
        assert self.ndim == 2, "Dimensions = 2 required, this is matrix transposition" 
        return self.comm(sops.Permute, self, axes=(1, 0))

