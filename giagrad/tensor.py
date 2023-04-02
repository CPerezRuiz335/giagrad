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

        Returns
        -------
        ndarray | float:
            the result of applying that operator to :paramref:`*tensors`'s data,
            i.g. float for reduction operators in some cases or a new ndarray
        Context:
            an instance to be passed to Tensor through :func:`~giagrad.Tensor.comm`
        """
        raise NotImplementedError(f"forward not implemented for {type(cls)}")
    
    @abstractmethod
    def backward(self, partial: NDArray):
        """
        Backpropagate from child Tensor created with :func:`~giagrad.Tensor.comm`.
        
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
            data, 
            requires_grad: bool = False, 
            context: Optional[Context] = None, 
            name: str = '',
            dtype=np.float32):
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
    def zeros(self) -> Tensor: 
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

    def ones(self) -> Tensor: 
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

    def constant(self, fill_value) -> Tensor: 
        """
        Fills Tensor data with a constant value. 
    
        Parameters
        ----------
        fill_value: Scalar
            the value to fill the output Tensor with

        Examples
        --------
            >>> Tensor.empty(2, 3).constant(2.71828)                                                                                           
            tensor: [[2.71828 2.71828 2.71828]
                     [2.71828 2.71828 2.71828]]  
        """
        self.data = np.full_like(self.data, fill_value=fill_value)
        return self
        
    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> Tensor: 
        r"""Fills Tensor data with values drawn from the normal
        distribution :math:`\mathcal{N}(\text{mu}, \sigma^2)`.

        Parameters
        ----------
        mu: float
            the mean of the normal distribution
        sigma: float
            the standard deviation of the normal distribution

        Examples
        --------
        >>> Tensor.empty(3, 3).normal()
        """
        init.normal(self, mu, sigma)
        return self

    def uniform(self, a: float = 0.0, b: float = 1.0) -> Tensor:
        r"""Fills Tensor data with values drawn from the uniform
        distribution :math:`\mathcal{U}(a, b)`.

        Parameters
        ----------
        a: float
            the lower bound of the uniform distribution
        b: float
            the upper bound of the uniform distribution

        Examples
        --------
        >>> Tensor.empty(3, 3).uniform()
        """ 
        init.uniform(self, a, b)
        return self

    def dirac(self, groups: int = 1) -> Tensor: 
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

    def xavier_uniform(self, gain: float = 1.0) -> Tensor: 
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
        >>> from giagrad import calculate_gain
        >>> Tensor.empty(3, 5).xavier_uniform(gain=calculate_gain('relu'))
        """
        init.xavier_uniform(self, gain=gain)
        return self
    
    def xavier_normal(self, gain: float = 1.0) -> Tensor: 
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
        >>> from giagrad import calculate_gain
        >>> Tensor.empty(3, 5).xavier_normal(gain=calculate_gain('relu'))
        """
        init.xavier_normal(self, gain=gain)
        return self    

    def kaiming_uniform(
        self, neg_slope: float = 0.0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'
    ) -> Tensor: 
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
    ) -> Tensor: 
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

    def sparse(self, sparsity: float, sigma=0.01) -> Tensor: 
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

    def orthogonal(self, gain: float = 1.0) -> Tensor:
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

        Examples
        --------
            >>> Tensor.empty(3, 5).orthogonal()
        """ 
        init.orthogonal(self, gain)
        return self

    ### MATH ###
    @classmethod
    def comm(cls, operator: Context, *tensors, **kwargs) -> Tensor:
        """
        Returns a new instance of an autodifferentiable Tensor given a :class:`giagrad.tensor.Context` operator.

        ``comm`` creates a Tensor with the output of :func:`~giagrad.tensor.Context.forward`.

        Parameters
        ----------
        *tensors: Any 
            everything that `numpy.array`_ constructor can accept. Internally ``comm``
            transforms any object in ``*tensors`` to a Tensor and passes it to 
            :func:`~giagrad.tensor.Context.forward`

        *kwargs: Any
            optional arguments passed to the :func:`~giagrad.tensor.Context.forward` method 
            of the ``operator`` parameter
    
        Examples
        --------
            >>> from giagrad.mlops import Softmax
            >>> t = Tensor.empty(2, 3).uniform(-1, 1)
            >>> t
            tensor: [[ 0.27639335  0.7524293   0.69203097]
                     [ 0.37772807 -0.9291505  -0.80418533]]
            >>> Tensor.comm(Softmax, t, axis=1)
            tensor: [[0.24242324 0.390224   0.36735278]
                     [0.6339727  0.17159334 0.19443396]] grad_fn: Softmax(axis = 1)

        .. _numpy.array: https://numpy.org/doc/stable/reference/generated/numpy.array.html
        """
        operands = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
        data, context = operator.forward(*operands, **kwargs)
        return cls(data, requires_grad=True, context=context)
    

    # ***** math functions (unary) ***** 
    def sqrt(self) -> Tensor: 
        """
        Returns a new tensor with the square-root of the elements of `data`. See :func:`~giagrad.Tensor.pow`.
        """
        return self.pow(0.5)

    def square(self) -> Tensor: 
        """
        Returns a new tensor with the square of the elements of `data`. See :func:`~giagrad.Tensor.pow`.
        """
        return self.pow(2)

    def exp(self) -> Tensor: 
        r"""
        Returns a new tensor with the exponential of the elements of `data`.

        .. math::
            out_i = \exp^{data_i}

        Examples
        --------
            >>> Tensor([0, 0.6931471805599453]).exp()
            tensor: [1. 2.] grad_fn: Exp
        """
        return Tensor.comm(mops.Exp, self)
    
    def log(self) -> Tensor: 
        r"""
        Returns a new tensor with the natural logarithm of the elements of `data`.

        .. math::
            out_i = \log_e(data_i)

        Examples
        --------
            >>> t = Tensor.empty(3).uniform() * 1e4
            >>> t
            tensor: [9553.524  3221.3936 6511.507 ] grad_fn: Mul
            >>> t.log()
            tensor: [7.650997 8.125444 8.514212] grad_fn: Ln
        """
        return Tensor.comm(mops.Log, self)

    def reciprocal(self) -> Tensor: 
        r"""
        Returns a new tensor with the reciprocal of the elements of `data`.

        .. math::
            out_i = \frac{1}{data_i}

        Examples
        --------
            >>> t = Tensor.empty(3).uniform() 
            >>> t
            tensor: [0.00142364 0.8617358  0.30606526]
            >>> t.reciprocal()
            tensor: [702.4239      1.1604484   3.267277 ] grad_fn: Reciprocal
        """
        return Tensor.comm(mops.Reciprocal, self)

    def abs(self) -> Tensor: 
        r"""
        Returns a new tensor with the absolute value of the elements of `data`.

        .. math::
            out_i = \lvert data_i \rvert
    
        Examples
        --------
            >>> Tensor([-1, -2, -3]).abs()
            tensor: [1. 2. 3.] grad_fn: Abs
        """
        return Tensor.comm(mops.Abs, self) 

    # ***** math functions (binary) *****
    def add(self, other) -> Tensor: 
        """
        Returns a new tensor with the sum of `data` and ``other``.
        
        Parameters
        ----------
        other: Tensor | ndarray | Scalar
            the number or object to add to `data`
        """
        return self.__add__(other)
    
    def sub(self, other) -> Tensor: 
        """
        Returns a new tensor with the substraction of ``other`` from `data`.
        
        Parameters
        ----------
        other: Tensor | ndarray | Scalar
            the number or object to substract from `data`
        """
        return self.__sub__(other)

    def mul(self, other) -> Tensor: 
        """
        Returns a new tensor with the multiplication of `data` to ``other``.
        
        Parameters
        ----------
        other: Tensor | ndarray | Scalar
            the number or object that multiplies `data`
        """
        return self.__mul__(other)

    def pow(self, other) -> Tensor: 
        """
        Returns a new tensor with `data` raised to the power of ``other``.
        
        Parameters
        ----------
        other: Tensor | ndarray | Scalar
            the number or object that `data` is raised to
        """
        return self.__pow__(x)

    def matmul(self, other) -> Tensor: 
        """
        Returns a new tensor with the matrix multiplication of `data` and ``other``.
    
        Parameters
        ----------
        other: Tensor | ndarray | Scalar
            the number or object that `data` is multiplied to from the left-hand side
        """
        return self.__matmul__(other)
    
    def div(self, other) -> Tensor: 
        """
        Returns a new tensor with the division of `data` to ``other``.
        
        Parameters
        ----------
        other: Tensor | ndarray | Scalar
            the number or object that divides `data` 
        """
        return self.__truediv__(other)

    def __neg__(self): return 0.0-self # Tensor.comm(mops.Mul, self, -1)
    # TODO
    def clip(self, min_, max_): raise NotImplementedError() # ((self-min_).relu()+min_) - (self-max_).relu()
    def sign(self): raise NotImplementedError() # return self / (self.abs() + 1e-10)

    # ***** activation functions (unary) ***** 
    def relu(self) -> Tensor: 
        r"""
        Applies the Rectified Linear Unit (ReLU) function element-wise. See `ReLU`_.

        .. math::
            out_i = \max(0, data)

        .. _ReLU: https://paperswithcode.com/method/relu
        """
        return Tensor.comm(mlops.ReLU, self) 

    def sigmoid(self) -> Tensor: 
        r"""
        Returns a new Tensor with element-wise sigmoid function. See `sigmoid`_.

        For numerical stability sigmoid function is computed with `numpy.logaddexp`_.

        .. math::
            out_i = \frac{1}{(1 + \exp(-data_i))}

        .. _numpy.logaddexp: https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html
        .. _sigmoid: https://paperswithcode.com/method/sigmoid-activation
        """
        return Tensor.comm(mlops.Sigmoid, self) 

    def elu(self, alpha: float = 1.0) -> Tensor: 
        r"""
        Creates a new Tensor applying Exponential Linear Unit (ELU) function to `data`. See `ELU`_.
        
        .. math::
            out_i =
            \begin{cases} 
                data_i \ \ if \ \ data_i > 0 \\ 
                \text{alpha}(\exp(data_i) - 1) \ \ if \ \ x \leq 0 \\
            \end{cases}
            
        .. _ELU: https://paperswithcode.com/method/elu
        
        Parameters
        ----------
        alpha: float
            the :math:`\alpha` value for the ELU formulation
        """
        return Tensor.comm(mlops.ELU, self, alpha=alpha) 

    def silu(self, beta: float = 1.0) -> Tensor: 
        r"""
        Returns a new Tensor with element-wise Sigmoid-Weighted Linear Unit (SiLU) function,
        also called Swish. See `Swish`_.
    
        For numerical stability SiLU is computed with `numpy.logaddexp`_.

        .. math::
            out_i = \frac{data_i}{(1 + \exp(\text{beta} \cdot -data_i))} 
        
        .. _Swish: https://paperswithcode.com/method/swish
        .. _numpy.logaddexp: https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html
        
        Parameters
        ----------
        beta: float
            hyperparameter for Swish formulation.
        """
        return Tensor.comm(mlops.SiLU, self, beta=beta)

    def tanh(self) -> Tensor: 
        r"""
        Applies the Tanh function element-wise. See `Tanh`_.

        .. math::
            out_i = \frac{e^{data_i} - e^{-data_i}}{e^{data_i} + e^{-data_i}}

        .. _Tanh: https://paperswithcode.com/method/tanh-activation
        """
        return Tensor.comm(mlops.Tanh, self)

    def leakyrelu(self, neg_slope: float = 0.01) -> Tensor: 
        r"""
        Creates a new Tensor applying Leaky Rectified Linear Unit (Leaky ReLU) function to `data`. 
        See `Leaky ReLU`_ .
        
        .. math::
            out_i =
            \begin{cases} 
                data_i \ \ if \ \ data_i > 0 \\ 
                \text{neg_slope} \cdot data_i \ \ if \ \ x \leq 0 \\
            \end{cases}
        
        .. _Leaky ReLU: https://paperswithcode.com/method/elu

        Parameters
        ----------
        neg_slope: float
            controls de angle of the negative slope (which only affects negative input values)
        """
        return Tensor.comm(mlops.LeakyReLU, self, neg_slope=neg_slope)

    def softplus(self, beta: float = 1.0, limit: float = 20.0) -> Tensor: 
        r"""
        Applies the Softplus function element-wise. See `Softplus`_.

        For numerical stability the implementation reverts to the linear function when
        :math:`data_i \times \text{beta} > \text{limit}`.

        .. math::
            out_i = \frac{1}{\text{beta}} \cdot \log(1 + \exp(\text{beta} \cdot data_i))
    
        .. _Softplus: https://paperswithcode.com/method/softplus
        
        Parameters
        ----------
        beta: float
            the :math:`\beta` value for the Softplus formulation
        limit: float
            data times beta above this revert to a linear function
        """
        return Tensor.comm(mlops.Softplus, self, limit=limit, beta=beta)
    
    def quick_gelu(self) -> Tensor: 
        r"""
        Returns a new Tensor with element-wise Quick GELU. See `GELU`_.
        
        Quick GELU is an approximation of GELU through :func:`~giagrad.Tensor.silu` 
        with alpha = 1.702 to ease GELU's computational complexity. 
        
        .. _GELU: https://paperswithcode.com/method/gelu
        """
        return Tensor.comm(mlops.SiLU, self, beta=1.702)

    def gelu(self) -> Tensor: 
        r"""
        Creates a new Tensor applying Gaussina Error Linear Unit (Leaky ReLU) function to `data`. 
        See `GELU`_.
        
        .. math::
            out_i = data_i \ \Phi(data_i) 
                = data_i \cdot \frac{1}{2} \left[1 + \text{erf}(\frac{data_i}{\sqrt{2}})\right]
        
        Where :math:`\Phi` is the Gaussian cumulative distribution function.

        .. _GELU: https://paperswithcode.com/method/gelu
        """
        return Tensor.comm(mlops.GELU, self) 

    def relu6(self) -> Tensor: 
        r"""
        Applies a modified version of ReLU with maximum size of 6. See `ReLU6`_.

        .. math::
            out_i = \min(\max(0, x), 6)

        .. _ReLU6: https://paperswithcode.com/method/relu6
        """
        return Tensor.comm(mlops.ReLU6, self) 

    def mish(self, beta: float = 1.0, limit: float = 20.0) -> Tensor: 
        r"""
        Returns a new Tensor with element-wise Mish function. See `Mish`_.
        
        .. math::
            out_i = data_i \cdot \text{tanh} \, \text{softplus}(data_i)
        
        See :func:`~giagra.Tensor.softplus`

        .. _Mish: https://paperswithcode.com/method/mish

        Parameters
        ----------
        beta: float
            the :math:`\beta` value for the Softplus formulation
        limit: float
            data times beta above this revert to a linear function
        """
        return self * self.softplus().tanh()

    def hardswish(self) -> Tensor: 
        r"""
        Creates a new Tensor applying Hard Swish function to `data`. See `Hard Swish`_.
        
        .. math::
            out_i = data_i \, \frac{\text{ReLU6}(data_i + 3)}{6}
        
        .. _Hard Swish: https://paperswithcode.com/method/hard-swish
        """
        return Tensor.comm(mlops.Hardswish, self)

    def softmax(self, axis: int) -> Tensor: 
        r"""
        Applies Softmax function to every 1-D slice defined by ``axis``. See `Softmax`_.

        The elements of the n-dimensinal output Tensor will lie in de range :math:`[0, 1]`
        and sum to :math:`1`.
        
        Softmax for a one-dimensional slice is defined as:
        
        .. math::
            \text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)} 
    
        Parameters
        ----------
        axis: int
            the dimension along which Softmax will be computed (so every slice along dim will sum to 1)


        Examples
        --------
            >>> t = Tensor.empty(2, 3).uniform(-1, 1)
            >>> t
            tensor: [[ 0.27639335  0.7524293   0.69203097]
                     [ 0.37772807 -0.9291505  -0.80418533]]
            >>> t.softmax(axis=1)
            tensor: [[0.24242324 0.390224   0.36735278]
                     [0.6339727  0.17159334 0.19443396]] grad_fn: Softmax(axis = 1)

        .. _Softmax: https://paperswithcode.com/method/softmax
        """
        return Tensor.comm(mlops.Softmax, self, axis=axis)

    def log_softmax(self, axis: int) -> Tensor: 
        r"""
        Applies LogSoftmax function to every 1-D slice defined by ``axis``.

        LogSoftmax for a one-dimensional slice is defined as:
        
        .. math::
            \text{LogSoftmax}(x_i) = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)} \right)

        Parameters
        ----------
        axis: int
            the dimension along which LogSoftmax will be computed

        Examples
        --------
            >>> t = Tensor.empty(2, 3).uniform(-1, 1)
            >>> t
            tensor: [[-0.07469178  0.7226724   0.98966014]
                     [-0.01990889 -0.4521888   0.26520386]]
            >>> t.softmax(axis=1)
            tensor: [[-0.72091377 -0.26915795 -0.39513725]
                     [-0.6661309  -1.4440191  -1.1195936 ]] grad_fn: LogSoftmax(axis = 0)
        """
        return Tensor.comm(mlops.LogSoftmax, self, axis=axis)

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

    # ***** shape functions (reduction) *****
    # this operators create views
    def permute(self, axes=None): return self.comm(sops.Permute, self, axes=axes)
    def transpose(self, dim0, dim1): return self.comm(sops.Permute, self, axes=(dim1, dim0))
    @property
    def T(self): 
        """Returns a transposed view of a 2 dimensional Tensor."""
        assert self.ndim == 2, "Dimensions = 2 required, this is matrix transposition" 
        return self.comm(sops.Permute, self, axes=(1, 0))

