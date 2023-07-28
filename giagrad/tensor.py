from __future__ import annotations
from itertools import chain
from math import prod
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Optional, Literal, Type, Union, Set, Any

import numpy as np 
from numpy.typing import NDArray

class Function(ABC):
    __slots__ = 'parents', '_name'

    def __init__(self):
        self.parents = []
        self._name = type(self).__name__

    def save_for_backward(self, *tensors):
        """
        Saves parent tensors in :attr:`parents` for backward pass.

        Parameters
        ----------
        *tensors: Tensor, ...
            The input tensors of :meth:`~giagrad.Function.forward`.
        """
        assert all(isinstance(t, Tensor) for t in tensors), \
        "parents must not contain other types than Tensor"
        self.parents.extend(tensors)

    @abstractmethod
    def forward(self, *tensors, **kwargs) -> Union[NDArray, float]:
        """
        Makes forward pass.

        Parameters
        ----------
        *tensors: Tensor
            A variable number of tensors, e.g. two for binary operations
            such as :func:`~giagrad.Tensor.matmul`.

        **kwargs: 
            Optional arguments if needed.

        Returns
        -------
        ndarray or float:
            The result of applying that operation to :paramref:`*tensors`'s 
            data, i.g. float for reduction operations in some cases or a
            new ndarray.
        """
        raise NotImplementedError(f"forward not implemented for {type(self)}")
    
    @abstractmethod
    def backward(self, partial: NDArray):
        """
        Backpropagate from child tensor created with :func:`~giagrad.Tensor.comm`.
        
        Updates :attr:`~parents` gradient through chain rule. This 
        method is the extension of :func:`~giagrad.Tensor.backward` for 
        a concrete operation.

        Parameters
        ----------
        partial: ndarray
            Defines the partial derivative of the loss function with 
            respect to the child Tensor, the one created with 
            :func:`~giagrad.tensor.Function.forward`.
        """
        raise NotImplementedError(f"backward not implemented for {type(self)}")

    def __str__(self): 
        return self._name 

import giagrad.shapeops as sops
import giagrad.mathops as mops
import giagrad.reductionops as rops
import giagrad.mlops as mlops
import giagrad.otherops as oops
import giagrad.initializers as init

class Tensor:
    __array_ufunc__ = None # tell numpy to trust Tensor to make __r***__ method
    __slots__ = 'data', 'grad', 'fn', 'requires_grad', 'name'

    def __init__(
        self, 
        data, 
        requires_grad: bool = False, 
        fn: Optional[Function] = None, 
        name: str = '',
        dtype=np.float32
    ):
        self.data = np.array(data, dtype=dtype) if not isinstance(data, np.ndarray) else data
        self.grad = np.zeros(self.data.shape, dtype=self.data.dtype) if requires_grad else None
        self.fn = fn
        self.requires_grad = requires_grad
        self.name = name
    
    # ***** backprop *****
    def backward(self, retain_graph=False):
        """
        Computes the gradient of all preceeding tensors.
        
        The graph is differentiated using the chain rule. Whether it is
        scalar or non-scalar (i.e. its data has more than one element), 
        gradient is set to ones and backpropagated.

        This function accumulates gradients in every preceeding tensor, 
        you might need to zero .grad attributes or set them to None 
        before calling it. 

        Parameters
        ----------
        retain_graph: bool, default: False
            If ``False`` the graph used to compute the grads will be freed.
        """
        topo = []
        visited = set([self])    
        def build_topo(tensor: Tensor):
            if (function := tensor.fn):
                for t in function.parents:
                    if not t.requires_grad:
                        continue
                    if t not in visited:
                        visited.add(t)
                        build_topo(t)
                topo.append(tensor)

        build_topo(self)

        # chain rule 
        self.grad = np.ones_like(self.data) # dL/dL = 1
        for tensor in reversed(topo):
            tensor.fn.backward(tensor.grad)
            if not retain_graph: 
                tensor.fn = None 

        del topo, visited # outsmart gargabe collector

    # ***** helpers *****
    @property
    def shape(self) -> Tuple[int, ...]: 
        """
        Tuple of tensor dimensions.

        Unlike numpy.ndarray.shape it can not be used to 
        reshape inplace.
        """
        return self.data.shape
    
    @property
    def dtype(self) -> type: 
        """Data-type of the tensor."""
        return self.data.dtype

    @property
    def size(self) -> int: 
        """Size of the tensor."""
        return self.data.size

    @property
    def ndim(self) -> int: 
        """Number of dimensions."""
        return self.data.ndim

    def no_grad(self) -> Tensor: 
        """Makes tensor not autodifferentiable."""
        self.requires_grad = False
        self.grad = None
        return self

    def requires_grad_(self) -> Tensor:
        """Makes tensor autodifferentiable.""" 
        self.requires_grad = True
        if self.grad is None:
            self.grad = np.zeros(self.data.shape, dtype=self.data.dtype)
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            'tensor: ' 
            + np.array2string(self.data, prefix='tensor: ', precision=4) 
            + (f" fn: {self.fn}" if self.fn else '') 
            + (f", name: {self.name}" if self.name else '')
        )
    # ***** initializers in-place*****
    @classmethod
    def empty(cls, *shape, dtype=np.float32, **kwargs) -> Tensor: 
        r"""
        Creates a tensor filled with uninitialized data. 

        Datatype is ``np.float32`` by default.
    
        Parameters
        ----------
        shape: int, ...
            A variable number of integers defining the shape of the 
            output tensor.
        \*\*kwargs:
            Parameters passed to the Tensor class initializer.
    
        Examples
        --------
        >>> Tensor.empty(2, 3, requires_grad=True, dtype=np.float64)
        tensor: [[4.67662529e-310 0.00000000e+000 4.67596337e-310]
                 [6.94592882e-310 6.94611561e-310 6.94609055e-310]]    
        """
        return cls(np.empty(shape, dtype=dtype), **kwargs)

    # in-place initializers
    def zeros(self) -> Tensor: 
        """
        Fills tensor data with zeros. 
    
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
        Fills tensor data with ones. 
    
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
        Fills tensor data with a constant value. 
    
        Parameters
        ----------
        fill_value: float
            The value to fill the tensor with.

        Examples
        --------
        >>> Tensor.empty(2, 3).constant(2.71828)                                                                                           
        tensor: [[2.71828 2.71828 2.71828]
                 [2.71828 2.71828 2.71828]]  
        """
        self.data = np.full_like(self.data, fill_value=fill_value)
        return self
        
    def normal(self, mu=0., std=1.) -> Tensor: 
        r"""
        Fills tensor data with values drawn from the normal
        distribution :math:`\mathcal{N}(\text{mu}, \text{std}^2)`.

        Parameters
        ----------
        mu: float
            Mean of the normal distribution.
        std: float
            The standard deviation of the normal distribution.

        Examples
        --------
        >>> Tensor.empty(3, 3).normal()
        """
        init.normal(self, mu, std)
        return self

    def uniform(self, a=0., b=1.) -> Tensor:
        r"""
        Fills Tensor data with values drawn from the uniform
        distribution :math:`\mathcal{U}(a, b)`.

        Parameters
        ----------
        a: float
            The lower bound of the uniform distribution.
        b: float
            The upper bound of the uniform distribution.

        Examples
        --------
        >>> Tensor.empty(3, 3).uniform()
        """ 
        init.uniform(self, a, b)
        return self

    def dirac(self, groups=1) -> Tensor: 
        r"""
        Fills the {3, 4, 5}-dimensional Tensor data with the Dirac 
        delta function. 

        Preserves the identity of the inputs in *Convolutional* layers, 
        where as many input channels are preserved as possible. In case
        of groups > 1, each group of channels preserves identity.

        Parameters
        ----------
        groups: int, default: 1
            Number of groups in the conv layer.
        
        Examples
        --------
        >>> Tensor.empty(3, 24, 5, 5).dirac(3)
        """
        init.dirac(self, groups=groups)
        return self

    def xavier_uniform(self, gain=1.) -> Tensor: 
        r"""
        Fills Tensor data with the also known Glorot uniform initialization.

        This methos is described in `Understanding the difficulty of 
        training deep feedforward neural networks` - Glorot, X. & Bengio, 
        Y. (2010), using a uniform distribution. Tensor data will have 
        values sampled from :math:`\mathcal{U}(-a, a)` where

        .. math::
            a = \text{gain} \times \sqrt{\frac{6}{\text{fan_in} + \text{fan_out}}}

        Parameters
        ----------
        gain: float
            An optional scaling factor.

        Examples
        --------
        >>> from giagrad import calculate_gain
        >>> Tensor.empty(3, 5).xavier_uniform(gain=calculate_gain('relu'))
        """
        init.xavier_uniform(self, gain=gain)
        return self
    
    def xavier_normal(self, gain=1.) -> Tensor: 
        r"""
        Fills Tensor data with the also known Glorot normal initialization.

        This method is described in `Understanding the difficulty of 
        training deep feedforward neural networks` - Glorot, X. & Bengio,
        Y. (2010), using a normal distribution. Tensor data will have 
        values sampled from :math:`\mathcal{N}(0, \sigma^2)` where

        .. math::
            \sigma = \text{gain} \times \sqrt{\frac{2}{\text{fan_in} + \text{fan_out}}}

        Parameters
        ----------
        gain: float
            An optional scaling factor.

        Examples
        --------
        >>> from giagrad import calculate_gain
        >>> Tensor.empty(3, 5).xavier_normal(gain=calculate_gain('relu'))
        """
        init.xavier_normal(self, gain=gain)
        return self    

    def kaiming_uniform(self, neg_slope=0., mode='fan_in', nonlinearity='leaky_relu') -> Tensor: 
        r"""
        Fills Tensor data with the also known He uniform initialization.

        Tensor data is filled with values according to the method 
        described in `Delving deep into rectifiers`_ using uniform 
        distribution. The resulting tensor will have values sampled 
        from :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

        .. math::
            \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan_mode}}}

        Parameters
        ----------
        neg_slope: float
            The negative slope of the rectifier used after this layer 
            (only used with `'leaky_relu'`).
        mode: str, default: 'fan_in'
            Either `'fan_in'` or `'fan_out'`. Choosing `'fan_in'`
            preserves the magnitude of the variance of the weights in 
            the forward pass. Choosing `'fan_out'` preserves the 
            magnitudes in the backwards pass.
        nonlinearity: str, default: 'leaky_relu'
            The non-linear function method name, recommended to use only 
            with `'relu'` or `'leaky_relu'`.

        Examples
        --------
        >>> Tensor.empty(3, 5).kaiming_uniform(mode='fan_in', nonlinearity='relu')

        .. _Delving deep into rectifiers: https://arxiv.org/abs/1502.01852
        """
        init.kaiming_uniform(self, neg_slope, mode, nonlinearity)
        return self

    def kaiming_normal(self, neg_slope=0., mode='fan_in', nonlinearity='leaky_relu') -> Tensor: 
        r"""
        Fills Tensor data with the also known He normal initialization.

        Tensor data is filled with values according to the method 
        described in `Delving deep into rectifiers`_ using normal 
        distribution. The resulting tensor will have values sampled 
        from :math:`\mathcal{N}(0, \sigma^2)` where

        .. math::
            \sigma = \frac{\text{gain}}{\sqrt{\text{fan_mode}}}

        Parameters
        ----------
        neg_slope: float
            The negative slope of the rectifier used after this layer 
            (only used with `'leaky_relu'`).
        mode: str, default: 'fan_in'
            Either `'fan_in'` or `'fan_out'`. Choosing `'fan_in'`
            preserves the magnitude of the variance of the weights in 
            the forward pass. Choosing `'fan_out'` preserves the 
            magnitudes in the backwards pass.
        nonlinearity: str, default: 'leaky_relu'
            The non-linear function method name,
            recommended to use only with `'relu'` or `'leaky_relu'`.

        Examples
        --------
        >>> Tensor.empty(3, 5).kaiming_normal(mode='fan_in', nonlinearity='relu')

        .. _Delving deep into rectifiers: https://arxiv.org/abs/1502.01852
        """
        init.kaiming_normal(self, neg_slope, mode, nonlinearity)
        return self    

    def sparse(self, sparsity, std=0.01) -> Tensor: 
        r"""
        Fills the 2D Tensor data as a sparse matrix.

        Non-zero elements will be drawn from the normal distribution
        :math:`\mathcal{N}(0, \text{sigma})`, as described in `Deep 
        learning via Hessian-free optimization`_ - Martens, J. (2010).

        Parameters
        ----------
        sparsity: float between [0, 1) 
            The fraction of elements in each column to be set to zero.
        std: float 
            The standard deviation of the normal distribution used to 
            generate the non-zero values.

        Examples
        --------
        >>> Tensor.empty(3, 5).sparse(sparsity=0.4, std=0.2)

        .. _Deep learning via Hessian-free optimization: https://dl.acm.org/doi/10.5555/3104322.3104416
        """
        init.sparse(self, sparsity, std) 
        return self

    def orthogonal(self, gain=1.) -> Tensor:
        r"""
        Fills Tensor data with a (semi) orthogonal matrix.

        Values are generated according to the method described in Exact 
        solutions to the nonlinear dynamics of learning in deep linear 
        neural networks - Saxe, A. et al. (2013). The Tensor must have
        at least 2 dimensions, and for Tensors with more than 2 
        dimensions the trailing dimensions are flattened.

        Parameters
        ----------
        gain: float
            Optional scaling factor.

        Examples
        --------
        >>> Tensor.empty(3, 5).orthogonal()
        """ 
        init.orthogonal(self, gain)
        return self

    ### MATH ###
    @classmethod
    def comm(cls, function: Function, *tensors) -> Tensor:
        """
        Returns a new instance of an autodifferentiable tensor given a 
        :class:`giagrad.tensor.Function`.

        ``comm`` creates a tensor with the output of 
        :func:`~giagrad.tensor.Function.forward`.
    
        For developer use.

        Parameters
        ----------
        *tensors: array_like, ... 
            Internally ``comm`` transforms any object in ``*tensors`` to 
            a Tensor and passes it to :func:`~giagrad.tensor.Function.forward`.

        *kwargs: 
            Optional arguments passed to the 
            :func:`~giagrad.tensor.Function.forward` method of the 
            ``fn`` parameter.
    
        Examples
        --------
        >>> from giagrad.mlops import Softmax
        >>> t = Tensor.empty(2, 3).uniform(-1, 1)
        >>> t
        tensor: [[ 0.27639335  0.7524293   0.69203097]
                 [ 0.37772807 -0.9291505  -0.80418533]]
        >>> Tensor.comm(Softmax(axis=1), t)
        tensor: [[0.24242324 0.390224   0.36735278]
                 [0.6339727  0.17159334 0.19443396]] fn: Softmax(axis=1)

        .. _numpy.array: https://numpy.org/doc/stable/reference/generated/numpy.array.html
        """
        operands = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
        data = function.forward(*operands)
        # NOTE: if no leaf tensor requires_grad, neither intermediate ones  
        requires_grad = any(t.requires_grad for t in operands)
        return cls(data, requires_grad=requires_grad, fn=function)
    

    # ***** math functions (unary) ***** 
    def sqrt(self) -> Tensor: 
        """
        Returns a new tensor with the square-root of the elements 
        of `data`.
        
        See Also
        --------
        :func:`~giagrad.Tensor.pow`.
        """
        return self.pow(0.5)

    def square(self) -> Tensor: 
        """
        Returns a new tensor with the square of the elements 
        of `data`. 
        
        See Also
        --------
        :func:`~giagrad.Tensor.pow`.
        """
        return self.pow(2)

    def exp(self) -> Tensor: 
        r"""
        Returns a new tensor with the exponential of the elements 
        of `data`.

        .. math::
            out_i = \exp^{data_i}

        Examples
        --------
        >>> Tensor([0, 0.6931471805599453]).exp()
        tensor: [1. 2.] fn: Exp
        """
        return Tensor.comm(mops.Exp(), self)
    
    def log(self) -> Tensor: 
        r"""
        Returns a new tensor with the natural logarithm of the elements 
        of `data`.

        .. math::
            out_i = \log_e(data_i)

        Examples
        --------
        >>> t = Tensor.empty(3).uniform() * 1e4
        >>> t
        tensor: [9553.524  3221.3936 6511.507 ] fn: Mul
        >>> t.log()
        tensor: [7.650997 8.125444 8.514212] fn: Ln
        """
        return Tensor.comm(mops.Log(), self)

    def reciprocal(self) -> Tensor: 
        r"""
        Returns a new tensor with the reciprocal of the elements 
        of `data`.

        .. math::
            out_i = \frac{1}{data_i}

        Examples
        --------
        >>> t = Tensor.empty(3).uniform() 
        >>> t
        tensor: [0.00142364 0.8617358  0.30606526]
        >>> t.reciprocal()
        tensor: [702.4239      1.1604484   3.267277 ] fn: Reciprocal
        """
        return Tensor.comm(mops.Reciprocal(), self)

    def abs(self) -> Tensor: 
        r"""
        Returns a new tensor with the absolute value of the elements 
        of `data`.

        .. math::
            out_i = \lvert data_i \rvert
    
        Examples
        --------
        >>> Tensor([-1, -2, -3]).abs()
        tensor: [1. 2. 3.] fn: Abs
        """
        return Tensor.comm(mops.Abs(), self) 

    # ***** math functions (binary) *****
    def add(self, other) -> Tensor: 
        """
        Returns a new tensor with the sum of `data` and ``other``.
        
        Parameters
        ----------
        other: array_like or float
            The number or object to add to `data`.
        """
        return self.__add__(other)
    
    def sub(self, other) -> Tensor: 
        """
        Returns a new tensor with the substraction of ``other`` 
        from `data`.
        
        Parameters
        ----------
        other: array_like or float
            The number or object to substract from `data`.
        """
        return self.__sub__(other)

    def mul(self, other) -> Tensor: 
        """
        Returns a new tensor with the multiplication of `data` 
        to ``other``.
        
        Parameters
        ----------
        other: array_like or float
            The number or object that multiplies `data`.
        """
        return self.__mul__(other)

    def pow(self, other) -> Tensor: 
        """
        Returns a new tensor with `data` raised to the power of ``other``.
        
        Parameters
        ----------
        other: array_like or float
            The number or object that `data` is raised to.
        """
        return self.__pow__(other)

    def matmul(self, other) -> Tensor: 
        """
        Returns a new tensor with the matrix multiplication 
        of `data` and ``other``.
    
        Parameters
        ----------
        other: array_like 
            The array_like object that `data` is multiplied to 
            from the left-hand side.
        """
        return self.__matmul__(other)
    
    def div(self, other) -> Tensor: 
        """
        Returns a new tensor with the division of `data` to ``other``.
        
        Parameters
        ----------
        other: array_like or float
            The number or object that divides `data`.
        """
        return self.__truediv__(other)

    def __neg__(self): return 0.0-self # Tensor.comm(mops.Mul, self, -1)
    # TODO
    def clip(self, min_, max_): raise NotImplementedError() # ((self-min_).relu()+min_) - (self-max_).relu()
    def sign(self): raise NotImplementedError() # return self / (self.abs() + 1e-10)

    # ***** activation functions (unary) ***** 
    def relu(self) -> Tensor: 
        r"""
        Applies the Rectified Linear Unit (ReLU) function element-wise. 
        See `ReLU`_.

        .. math::
            out_i = \max(0, data)
    
        Examples
        --------
        >>> t = Tensor.empty(2, 3).uniform(-1, 1)
        >>> t
        tensor: [[ 0.96863234  0.64852756 -0.52318954]
                 [-0.18809071 -0.48402452  0.86754996]]
        >>> t.relu()
        tensor: [[0.96863234 0.64852756 0.        ]
                 [0.         0.         0.86754996]] fn: ReLU

        .. _ReLU: https://paperswithcode.com/method/relu
        """
        return Tensor.comm(mlops.ReLU(), self) 

    def sigmoid(self) -> Tensor: 
        r"""
        Returns a new Tensor with element-wise sigmoid function. 
        See `sigmoid`_.

        For numerical stability sigmoid function is computed with 
        `numpy.logaddexp`_.

        .. math::
            out_i = \frac{1}{(1 + \exp(-data_i))}

        Examples
        --------
        >>> t = Tensor.empty(2, 3).uniform(-100, 100)
        >>> t
        tensor: [[-49.970577  35.522175 -14.944364]
                 [ 32.187164 -66.65264   48.01228 ]]
        >>> t.sigmoid()
        tensor: [[1.9863422e-22 1.0000000e+00 3.2340398e-07]
                 [1.0000000e+00 1.1301229e-29 1.0000000e+00]] fn: Sigmoid

        .. _numpy.logaddexp: https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html
        .. _sigmoid: https://paperswithcode.com/method/sigmoid-activation
        """
        return Tensor.comm(mlops.Sigmoid(), self) 

    def elu(self, alpha=1.) -> Tensor: 
        r"""
        Creates a new Tensor applying Exponential Linear Unit (ELU) 
        function to `data`. See `ELU`_.
        
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
            The :math:`\alpha` value for the ELU formulation.

        Examples
        --------
        >>> t = Tensor.empty(2, 3).uniform(-100, 100)
        >>> t
        tensor: [[-49.970577  35.522175 -14.944364]
                 [ 32.187164 -66.65264   48.01228 ]]
        >>> t.elu()
        tensor: [[-1.        35.522175  -0.9999997]
                 [32.187164  -1.        48.01228  ]] fn: ELU(alpha=1.0)
        """
        return Tensor.comm(mlops.ELU(alpha=alpha), self) 

    def silu(self, beta=1.) -> Tensor: 
        r"""
        Returns a new Tensor with element-wise Sigmoid-Weighted Linear 
        Unit (SiLU) function, also called Swish. See `Swish`_.
    
        For numerical stability SiLU is computed with `numpy.logaddexp`_.

        .. math::
            out_i = \frac{data_i}{(1 + \exp(\text{beta} \times -data_i))} 
        
        .. _Swish: https://paperswithcode.com/method/swish
        .. _numpy.logaddexp: https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html
        
        Parameters
        ----------
        beta: float
            Hyperparameter for Swish formulation.

        Examples
        --------
        >>> t = Tensor.empty(2, 3).uniform(-10, 10)
        >>> t
        tensor: [[ 5.4958744   0.13549101 -4.5210676 ]
                 [-1.7155124   5.2369795  -7.6546626 ]]
        >>> t.silu()
        tensor: [[ 5.4734135e+00  7.2327957e-02 -4.8648320e-02]
                 [-2.6153007e-01  5.2092857e+00 -3.6252895e-03]] fn: SiLU(beta=1.0)
        """
        return Tensor.comm(mlops.SiLU(beta=beta), self)

    def tanh(self) -> Tensor: 
        r"""
        Applies the Tanh function element-wise. See `Tanh`_.

        .. math::
            out_i = \frac{e^{data_i} - e^{-data_i}}{e^{data_i} + e^{-data_i}}

        Examples
        --------
        >>> t = Tensor.empty(2, 3).uniform(-8, 8)                                                                
        >>> t
        tensor: [[-0.42122853 -3.4285958   7.846644  ]
                 [ 0.7483299   6.6553855   3.3439522 ]]
        >>> t.tanh()                                                                                             
        tensor: [[-0.3979649  -0.9978985   0.9999997 ]
                 [ 0.6341515   0.99999666  0.9975113 ]] fn: tanh

        .. _Tanh: https://paperswithcode.com/method/tanh-activation
        """
        return Tensor.comm(mlops.Tanh(), self)

    def leakyrelu(self, neg_slope=0.01) -> Tensor: 
        r"""
        Creates a new Tensor applying Leaky Rectified Linear Unit 
        (Leaky ReLU) function to `data`. See `Leaky ReLU`_ .
        
        .. math::
            out_i =
            \begin{cases} 
                data_i \ \ if \ \ data_i > 0 \\ 
                \text{neg_slope} \times data_i \ \ if \ \ x \leq 0 \\
            \end{cases}
        
        .. _Leaky ReLU: https://paperswithcode.com/method/elu

        Parameters
        ----------
        neg_slope: float
            Controls de angle of the negative slope (which only affects 
            negative input values).

        Examples
        --------
        >>> t = Tensor.empty(2, 3, requires_grad=True).uniform(-1, 1)
        >>> t
        tensor: [[-0.83589154  0.8874637  -0.465633  ]
                 [-0.5879877   0.22095676 -0.0592072 ]]
        >>> d = t.leakyrelu(neg_slope=3)                                                                         
        >>> d
        tensor: [[-2.5076747   0.8874637  -1.396899  ]
                 [-1.7639632   0.22095676 -0.17762159]] fn: LeakyReLU(neg_slope=3)
        >>> d.backward()                                                                                         
        >>> t.grad
        array([[3., 1., 3.],
               [3., 1., 3.]], dtype=float32)
        """
        return Tensor.comm(mlops.LeakyReLU(neg_slope=neg_slope), self)

    def softplus(self, beta=1., limit=20.) -> Tensor: 
        r"""
        Applies the Softplus function element-wise. See `Softplus`_.

        For numerical stability the implementation reverts to the linear 
        function when :math:`data_i \times \text{beta} > \text{limit}`.

        .. math::
            out_i = \frac{1}{\text{beta}} \cdot \log(1 + \exp(\text{beta} \times data_i))
    
        .. _Softplus: https://paperswithcode.com/method/softplus
        
        Parameters
        ----------
        beta: float
            The :math:`\beta` value for the Softplus formulation.
        limit: float
            Data times beta above this reverts to a linear function.

        Examples
        --------
        >>> t = Tensor.empty(2, 3).uniform(-1, 1)                                                                
        >>> t                                                                                                    
        tensor: [[ 0.54631704 -0.703394    0.85786563]
                 [-0.24458279  0.23733494 -0.32190484]]
        >>> t.softplus(beta=5, limit=1)
        tensor: [[0.54631704 0.00585142 0.85786563]
                 [0.05160499 0.23733494 0.03646144]] fn: Softplus(lim=1, alpha=5)
        """
        return Tensor.comm(mlops.Softplus(limit=limit, beta=beta), self)
    
    def quick_gelu(self) -> Tensor: 
        r"""
        Returns a new Tensor with element-wise Quick GELU. See `GELU`_.
        
        Quick GELU is an approximation of GELU through 
        :func:`~giagrad.Tensor.silu` with alpha = 1.702 to 
        ease GELU's computational complexity. 
        
        Examples
        --------
        >>> t = Tensor.empty(2, 3).uniform(-1, 1)                                                       
        >>> t
        tensor: [[ 0.62271285  0.37412217 -0.6465454 ]
                 [-0.9013401  -0.02915052 -0.9814293 ]]
        >>> t.quick_gelu()
        tensor: [[ 0.4624659   0.2446833  -0.16141725]
                 [-0.15989538 -0.01421376 -0.15543076]] fn: QuickGELU

        .. _GELU: https://paperswithcode.com/method/gelu
        """
        return Tensor.comm(mlops.SiLU(beta=1.702), self)

    def gelu(self) -> Tensor: 
        r"""
        Creates a new Tensor applying Gaussina Error Linear Unit 
        (Leaky ReLU) function to `data`. See `GELU`_.
        
        .. math::
            out_i = data_i \ \Phi(data_i) 
                = data_i \times \frac{1}{2} \left[1 + \text{erf}(\frac{data_i}{\sqrt{2}})\right]
        
        Where :math:`\Phi` is the Gaussian cumulative distribution function.
    
        Examples
        --------
        >>> t = Tensor.empty(2, 3).uniform(-1, 1)                                                                
        >>> t                                                                                                    
        tensor: [[-0.42565832  0.8579072  -0.40772486]
                 [ 0.4038496   0.09953032 -0.6694602 ]]
        >>> t.gelu()                                                                                             
        tensor: [[-0.14268097  0.6901076  -0.1393431 ]
                 [ 0.26525608  0.05371065 -0.1684846 ]] fn: GELU

        .. _GELU: https://paperswithcode.com/method/gelu
        """
        return Tensor.comm(mlops.GELU(), self) 

    def relu6(self) -> Tensor: 
        r"""
        Applies a modified version of ReLU with maximum size of 6. 
        See `ReLU6`_.

        .. math::
            out_i = \min(\max(0, x), 6)
        
        Examples
        --------
        >>> t = Tensor.empty(2, 3).uniform(-1, 20)
        >>> t
        tensor: [[11.792983   -0.20050316 15.441884  ]
                 [ 3.5337465  13.230399    9.813518  ]]
        >>> t.relu6()
        tensor: [[6.        0.        6.       ]
                 [3.5337465 6.        6.       ]] fn: ReLU6

        .. _ReLU6: https://paperswithcode.com/method/relu6
        """
        return Tensor.comm(mlops.ReLU6(), self) 

    def mish(self, beta=1., limit=20.) -> Tensor: 
        r"""
        Returns a new Tensor with element-wise Mish function. 
        See `Mish`_.
        
        .. math::
            out_i = data_i \times \text{tanh} \left( \text{softplus}(data_i) \right)
        
        .. _Mish: https://paperswithcode.com/method/mish

        Parameters
        ----------
        beta: float
            The :math:`\beta` value for the Softplus formulation.
        limit: float
            Data times beta above limit reverts to a linear function in 
            Softplus formulation.
        
        See Also
        --------
        :func:`~giagrad.Tensor.softplus`

        Examples
        --------
        >>> t = Tensor.empty(2, 3).uniform(-5, 5)                                                                
        >>> t
        tensor: [[-1.3851491  1.2130666 -4.9049625]
                 [ 2.6859815 -4.845946   2.1385565]]
        >>> t.mish()                                                                                             
        tensor: [[-0.3043592   1.0920179  -0.03620962]
                 [ 2.6642     -0.03794033  2.0915585 ]] fn: Mish(beta=1.0, lim=20.0)
        """
        return Tensor.comm(mlops.Mish(beta=beta, limit=limit), self) 

    def hardswish(self) -> Tensor: 
        r"""
        Creates a new Tensor applying Hard Swish function to `data`. 
        See `Hard Swish`_.
        
        .. math::
            out_i = data_i \, \times \, \frac{\text{ReLU6}(data_i + 3)}{6}

        Examples
        --------
        >>> t = Tensor.empty(2, 4).uniform(-5, 5)                                                                 
        >>> t
        tensor: [[-4.0175104   3.993501   -1.0318986  -0.30065283]
                 [-2.4765007  -1.3878915   1.7888396   4.3194094 ]]
        >>> t.hardswish()                                                                 
        tensor: [[-0.          3.993501   -0.3384802  -0.13526106]
                 [-0.21607438 -0.37290528  1.4277442   4.3194094 ]] fn: Hardswish
        
        .. _Hard Swish: https://paperswithcode.com/method/hard-swish
        """
        return Tensor.comm(mlops.Hardswish(), self)

    def softmax(self, axis) -> Tensor: 
        r"""
        Applies Softmax function to every 1-D slice defined by ``axis``. 
        See `Softmax`_.

        The elements of the n-dimensinal output Tensor will lie in the 
        range :math:`[0, 1]` and sum to :math:`1` for the specified 1-D 
        slices defined by ``axis``.
        
        Softmax for a one-dimensional slice is defined as:
        
        .. math::
            \text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)} 
    
        Parameters
        ----------
        axis: int
            The dimension along which Softmax will be computed 
            (so every slice along axis will sum to 1).


        Examples
        --------
        >>> t = Tensor.empty(2, 3).uniform(-1, 1)
        >>> t
        tensor: [[ 0.27639335  0.7524293   0.69203097]
                 [ 0.37772807 -0.9291505  -0.80418533]]
        >>> t.softmax(axis=1)
        tensor: [[0.24242324 0.390224   0.36735278]
                 [0.6339727  0.17159334 0.19443396]] fn: Softmax(axis=1)

        .. _Softmax: https://paperswithcode.com/method/softmax
        """
        return Tensor.comm(mlops.Softmax(axis=axis), self)

    def log_softmax(self, axis: int) -> Tensor: 
        r"""
        Applies LogSoftmax function to every 1-D slice defined by ``axis``.

        LogSoftmax for a one-dimensional slice is defined as:
        
        .. math::
            \text{LogSoftmax}(x_i) = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)} \right)

        Parameters
        ----------
        axis: int
            The dimension along which LogSoftmax will be computed.

        Examples
        --------
        >>> t = Tensor.empty(2, 3).uniform(-1, 1)
        >>> t
        tensor: [[-0.07469178  0.7226724   0.98966014]
                 [-0.01990889 -0.4521888   0.26520386]]
        >>> t.softmax(axis=1)
        tensor: [[-0.72091377 -0.26915795 -0.39513725]
                 [-0.6661309  -1.4440191  -1.1195936 ]] fn: LogSoftmax(axis=0)
        """
        return Tensor.comm(mlops.LogSoftmax(axis=axis), self)

    # ***** math functions (binary) *****
    def __add__(self, x): return Tensor.comm(mops.Add(), self, x)
    def __radd__(self, x): return Tensor.comm(mops.Add(), x, self)
    def __sub__(self, x): return Tensor.comm(mops.Sub(), self, x)
    def __rsub__(self, x): return Tensor.comm(mops.Sub(), x, self)
    def __mul__(self, x): return Tensor.comm(mops.Mul(), self, x)
    def __rmul__(self, x): return Tensor.comm(mops.Mul(), x, self)
    def __pow__(self, x): return Tensor.comm(mops.Pow(), self, x)
    def __rpow__(self, x): return Tensor.comm(mops.Pow(), x, self)
    def __matmul__(self, x): return Tensor.comm(mops.Matmul(), self, x)
    def __rmatmul__(self, x): return Tensor.comm(mops.Matmul(), x, self)
    def __truediv__(self, x): return Tensor.comm(mops.Div(), self, x) 
    def __rtruediv__(self, x): return Tensor.comm(mops.Div(), x, self) 

    # ***** math functions autossign (i.e. a += b) *****
    def __iadd__(self, x): self.data += x.data if isinstance(x, Tensor) else x; return self
    def __isub__(self, x): self.data -= x.data if isinstance(x, Tensor) else x; return self
    def __imul__(self, x): self.data *= x.data if isinstance(x, Tensor) else x; return self
    def __ipow__(self, x): self.data **= x.data if isinstance(x, Tensor) else x; return self
    def __itruediv__(self, x): self.data /= x.data if isinstance(x, Tensor) else x; return self
    def __imatmul__(self, x): self.data = self.data @ x.data if isinstance(x, Tensor) else x; return self

    # ***** logical *****
    def __lt__(self, x): return Tensor(self.data  < x, dtype=np.bool_)  
    def __gt__(self, x): return Tensor(self.data  > x, dtype=np.bool_)  
    def __le__(self, x): return Tensor(self.data <= x, dtype=np.bool_)   
    def __eq__(self, x): return Tensor(self.data == x, dtype=np.bool_)        
    def __ne__(self, x): return Tensor(self.data != x, dtype=np.bool_)       
    def __ge__(self, x): return Tensor(self.data >= x, dtype=np.bool_)   
    # need __hash__ due to __eq__
    def __hash__(self): return hash((id(self), self.fn, self.requires_grad, self.name))
    def __setitem__(self, idx, value): 
        if self.requires_grad: 
            raise RuntimeError("Assignments only to non differentiable tensors")
        self.data[idx] = value

    # ***** math functions (reduction) *****
    def mean(self, axis=None, keepdims=False) -> Tensor: 
        r"""
        Returns the mean value of each 1-D slice of the tensor in the 
        given ``axis``, if ``axis`` is a list of dimensions, reduce 
        over all of them.

        If keepdims is True, the output tensor is of the same size as 
        input except in the ``axis`` where it is of size 1. Otherwise, 
        every ``axis`` is squeezed, leading to an output tensor with 
        fewer dimensions. If no ``axis`` is supplied all data is 
        reduced to a scalar value.

        Parameters
        ----------
        axis: (int, ...) or int or None, default: None
            The dimension or dimension to reduce. If None, mean reduces 
            all dimensions.
        keepdims: bool, default: False
            Whether te output tensor should retain the reduced dimensions.

        Examples
        --------
        >>> t = Tensor(np.arange(12).reshape((2,2,3)))
        >>> t
        tensor: [[[ 0.  1.  2.]
                  [ 3.  4.  5.]]
        ...
                 [[ 6.  7.  8.]
                  [ 9. 10. 11.]]]
        >>> t.mean(axis=(0, 1), keepdims=True)                                      
        tensor: [[[4.5 5.5 6.5]]] fn: Mean(axis=(0, 1))
        """
        return Tensor.comm(rops.Mean(axis=axis, keepdims=keepdims), self)
    
    def sum(self, axis=None, keepdims=False): 
        r"""
        Returns the sum of each 1-D slice of the tensor in the given 
        ``axis``, if ``axis`` is a list of dimensions, reduce over all 
        of them.

        If keepdims is True, the output tensor is of the same size as 
        input except in the ``axis`` where it is of size 1. Otherwise, 
        every ``axis`` is squeezed, leading to an output tensor with 
        fewer dimensions. If no ``axis`` is supplied all data is reduced
        to a scalar value.

        Parameters
        ----------
        axis: (int, ...) or int or None, default: None
            The dimension or dimension to reduce. If None, sum reduces 
            all dimensions.
        keepdims: bool, default: False
            Whether te output tensor should retain the reduced dimensions.

        Examples
        --------
        >>> t = Tensor.empty(2, 3, 4, dtype=int).uniform(0, 5)                          
        >>> t
        tensor: [[[2 0 0 3]
                  [0 2 1 4]
                  [4 0 0 2]]
        ...
                 [[3 1 4 0]
                  [3 3 4 3]
                  [4 0 1 0]]]
        >>> t.sum(axis=2, keepdims=True)                           
        tensor: [[[ 5.]
                  [ 7.]
                  [ 6.]]
        ...
                 [[ 8.]
                  [13.]
                  [ 5.]]] fn: Sum(axis=2)
        """
        return Tensor.comm(rops.Sum(axis=axis, keepdims=keepdims), self)
    
    def max(self, axis=None, keepdims=False): 
        r"""
        Returns the maximum value of each 1-D slice of the tensor in the 
        given ``axis``, if ``axis`` is a list of dimensions, reduce over 
        all of them.

        If keepdims is True, the output tensor is of the same size as 
        input except in the ``axis`` where it is of size 1. Otherwise, 
        every ``axis`` is squeezed, leading to an output tensor with 
        fewer dimensions. If no ``axis`` is supplied all data is reduced
        to a scalar value.

        Parameters
        ----------
        axis: (int, ...) or int or None, default: None
            The dimension or dimension to reduce. If None, max reduces 
            all dimensions.
        keepdims: bool, default: False
            Whether te output tensor should retain the reduced dimensions.

        Examples
        --------
        >>> t = Tensor.empty(2, 3, 4, dtype=np.int8).uniform(0, 100)
        >>> t
        tensor: [[[54 83 83 67]
                  [81 64 76 51]
                  [76 98 58 28]]
        ...
                 [[64 91 59 48]
                  [70 41 16 33]
                  [27 44 17 70]]]
        >>> t.max(axis=(1, 2))                                                          
        tensor: [98. 91.] fn: Max(axis=(1, 2))
        """
        return Tensor.comm(rops.MinMax(axis=axis, keepdims=keepdims, fn=np.max), self)

    def min(self, axis=None, keepdims=False): 
        r"""
        Returns the minimum value of each 1-D slice of the tensor in the 
        given ``axis``, if ``axis`` is a list of dimensions, reduce over 
        all of them.

        If keepdims is True, the output tensor is of the same size as 
        input except in the ``axis`` where it is of size 1. Otherwise, 
        every ``axis`` is squeezed, leading to an output tensor with 
        fewer dimensions. If no ``axis`` is supplied all data is reduced 
        to a scalar value.

        Parameters
        ----------
        axis: (int, ...) or int or None, default: None
            The dimension or dimension to reduce. If None, min reduces 
            all dimensions.
        keepdims: bool, default: False
            Whether te output tensor should retain the reduced dimensions.

        Examples
        --------
        >>> t = Tensor.empty(2, 3, 4, dtype=np.int8).uniform(0, 20)                     
        >>> t
        tensor: [[[ 3 14 15  7]
                  [18  9 11 18]
                  [16 17 14  9]]
        ...
                 [[ 5  3 12 18]
                  [15 11 15  1]
                  [13  2  2 10]]]
        >>> t.min(axis=2, keepdims=True)                                                
        tensor: [[[3.]
                  [9.]
                  [9.]]
        ...
                 [[3.]
                  [1.]
                  [2.]]] fn: Min(axis=2)
        """
        return Tensor.comm(rops.MinMax(axis=axis, keepdims=keepdims, fn=np.min), self)

    # ***** shape functions *****
    def reshape(self, *newshape) -> Tensor:
        """
        Returns a new tensor with shape equals ``newshape``.
        
        When possible, the returned tensor will be a view of the input.
        Otherwise, it will be a copy. Contiguous inputs and inputs with
        compatible strides can be reshaped without copying, but you 
        should not depend on the copying vs. viewing behavior.

        Parameters
        ----------
        *newshape: list of ints
            The new shape should be compatible with the original shape. 
            If an integer, then the result will be a 1-D array of that 
            length. One shape dimension can be -1. In this case, the 
            value is inferred from the length of the array and remaining 
            dimensions.

        Examples
        --------
        >>> a = Tensor.empty(6).ones()                                                                           
        >>> a.reshape(2, 3)                                                                                      
        tensor: [[1. 1. 1.]
                 [1. 1. 1.]] fn: Reshape
        >>> b = Tensor.empty(2, 2, 2).zeros()                                                                    
        >>> b.reshape(2, -1)
        tensor: [[0. 0. 0. 0.]
                 [0. 0. 0. 0.]] fn: Reshape
        """
        return Tensor.comm(sops.Reshape(newshape), self)

    def permute(self, axes=None): 
        """
        Returns a view of the original tensor with its ``axes`` permuted.

        ``permute`` uses `numpy.transpose`_, the following documentation 
        is adapted.

        For a 1-D tensor, this returns an unchanged view of the original 
        tensor, as a transposed vector is simply the same vector. 
        To convert a 1-D tensor into a 2-D tensor vector, an additional 
        dimension must be added, e.g., tensor.unsqueeze(axis=0) achieves 
        this, as does Tensor[:, None]. 

        For a 2-D tensor, this is the standard matrix transpose. For an 
        n-D tensor, if axes are given, their order indicates how the 
        axes are permuted (see Examples). If axes are not provided, then 
        tensor.permute().shape == tensor.shape[::-1].

        
        See Also
        --------
        :meth:`giagrad.Tensor.swapaxes`
            For swaping only two axes.
        :func:`numpy.transpose`

        Parameters
        ----------
        axes: tuple or list of ints, optional
            If specified, it must be a tuple or list which contains a 
            permutation of [0,1,...,N-1] where N is the number of axes 
            of the original tensor. The **i**’th axis of the returned 
            tensor will correspond to the axis numbered ``axes[i]`` of 
            the input. 
            If not specified, defaults to ``range(tensor.ndim)[::-1]``, 
            which reverses the order of the axes.

        Examples
        --------
        >>> t = Tensor.empty(1, 2, 3, 2, dtype=int).uniform(-5, 5)                    
        >>> t
        tensor: [[[[ 1  0]
                   [-3  4]
                   [ 3  3]]
        ...
                  [[ 3 -4]
                   [-3  1]
                   [-2  3]]]]
        
        Note that axes has the same lenght as :attr:`giagrad.Tensor.ndim`.

        >>> t.permute(axes=(1, 2, 3, 0))                                              
        tensor: [[[[ 1]
                   [ 0]]
        ...
                  [[-3]
                   [ 4]]
        ...
                  [[ 3]
                   [ 3]]]
        ...
        ...
                 [[[ 3]
                   [-4]]
        ...
                  [[-3]
                   [ 1]]
        ...
                  [[-2]
                   [ 3]]]] fn: Permute(axes = (1, 2, 3, 0))
        >>> t.permute(axes=(1, 2, 3, 0)).shape
        (2, 3, 2, 1)

        .. _numpy.transpose: https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
        """
        return self.comm(sops.Permute(axes=axes), self)

    def swapaxes(self, axis0, axis1):
        """
        Permutes two specific axes.

        Note
        ----
        The returned tensor shares the storage with the input tensor, 
        so changing the contents of one will change the contents of the 
        other.
        
        See Also
        --------
        :attr:`giagrad.Tensor.T`, :meth:`giagrad.Tensor.permute`

        Parameters
        ----------
        axis0: int
            First axis.
        axis1: int
            Second axis.

        Examples
        --------
        >>> t = Tensor.empty(1, 2, 3, 2, dtype=int).uniform(-100, 100)                
        >>> t                                                                         
        tensor: [[[[-91  22]
                   [ 54 -47]
                   [ 21 -88]]
        ...
                  [[  3 -78]
                   [ 34  68]
                   [-51  29]]]]
        >>> t.swapaxes(2, 3)                                                           
        tensor: [[[[-91  54  21]
                   [ 22 -47 -88]]
        ...
                  [[  3  34 -51]
                   [-78  68  29]]]] fn: Swapaxes(2, 3)
        """
        return self.comm(sops.Swapaxes(axis0, axis1), self)

    @property
    def T(self): 
        """Returns a transposed view of a 2 dimensional Tensor."""
        assert self.ndim == 2, "Dimensions = 2 required, this is matrix transposition" 
        return self.swapaxes(0, 1)

    def __getitem__(self, idx):
        return Tensor.comm(sops.Getitem(idx=idx), self)

    def pad(self, *padding, mode: str = 'constant', **kwargs):
        """
        Pads tensor.

        Padding size specified by ``*padding`` maps every argument 
        starting from the rightmost axis. If ``*padding`` is a single 
        int ``N`` it will be interpreted as if "before" and "after" 
        padding for the last axis is symmetric, i.e. ``(N_before, N_after)``. 
        If a tuple of two integers is supplied, it will be interpreted 
        as ``(N_before, N_after)`` padding.

        Padding ``mode`` has the same options as `numpy.pad`_.

        See Also
        --------
        :func:`numpy.pad`

        Parameters
        ----------
        *padding: int or (int, int)
            Number of values padded to the edges of the rightmost axes.
        mode: str, default: 'constant'
            Padding mode defined by `numpy.pad`_.
        **kwargs:
            Optional arguments passed to `numpy.pad`_.

        Examples
        --------
        >>> t = Tensor.empty(2, 2, 3, dtype=int).uniform(-5, 5)
        >>> t
        tensor: [[[ 0 -1  0]
                  [ 0  2  0]]
        ...
                 [[-1 -2  0]
                  [-3 -1  3]]]
        
        A single int padds the last axis with before and after 
        symmetrically:

        >>> t.pad(2)
        tensor: [[[ 0  0  0 -1  0  0  0]
                  [ 0  0  0  2  0  0  0]]
        ...
                 [[ 0  0 -1 -2  0  0  0]
                  [ 0  0 -3 -1  3  0  0]]] fn: ConstantPad
        >>> t.pad((1, 0), 2, (1, 3))
        tensor: [[[ 0  0  0  0  0  0  0]
                  [ 0  0  0  0  0  0  0]
                  [ 0  0  0  0  0  0  0]
                  [ 0  0  0  0  0  0  0]
                  [ 0  0  0  0  0  0  0]
                  [ 0  0  0  0  0  0  0]]
        ...
                 [[ 0  0  0  0  0  0  0]
                  [ 0  0  0  0  0  0  0]
                  [ 0  0 -1  0  0  0  0]
                  [ 0  0  2  0  0  0  0]
                  [ 0  0  0  0  0  0  0]
                  [ 0  0  0  0  0  0  0]]
        ...
                 [[ 0  0  0  0  0  0  0]
                  [ 0  0  0  0  0  0  0]
                  [ 0 -1 -2  0  0  0  0]
                  [ 0 -3 -1  3  0  0  0]
                  [ 0  0  0  0  0  0  0]
                  [ 0  0  0  0  0  0  0]]] fn: ConstantPad
        
        .. _numpy.pad: 
            https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        """
        all_tup = chain(*(i if isinstance(i, tuple) else (i,) for i in padding))
        if np.sum(tuple(all_tup)) == 0:
            return self
        return Tensor.comm(sops.Pad(padding, mode, **kwargs), self)

    def squeeze(self, axis=None):
        """
        Remove axes of length one.

        For example, a tensor of shape :math:`(1, N_1, N_2, 1, N_3, 1)` 
        will be reshaped into :math:`(N_1, N_2, N_3)` if no axis is 
        inputed. Specific axis with length one can be removed either 
        passing an int or a tuple or ints.
        
        Note
        ----
        The returned tensor shares the storage with the input tensor, so 
        changing the contents of one will change the contents of the other.

        Warning
        -------
        If the tensor has a batch dimension of size 1, then ``squeeze`` 
        will also remove the batch dimension, which can lead to unexpected 
        errors. Consider specifying only the dims you wish to be squeezed.

        See Also
        --------
        :meth:`giagrad.Tensor.unsqueeze`
        
        Parameters
        ----------
        axis: (int, ...) or int, optional
            By default removes all axes of length one, if tuple or int 
            passed those axes will be removed.
    
        Examples
        --------
        >>> t = Tensor.empty(1, 2, 1, 2, 1).uniform()                                             
        >>> t
        tensor: [[[[[0.16217469]
                    [0.8090288 ]]]
        ...
        ...
                  [[[0.7216649 ]
                    [0.6690301 ]]]]]
        >>> t.squeeze()
        tensor: [[0.16217469 0.8090288 ]
                 [0.7216649  0.6690301 ]] fn: Squeeze(axis = None)
        >>> t.squeeze().shape
        (2, 2)
        """
        return Tensor.comm(sops.Squeeze(axis), self)

    def unsqueeze(self, axis):
        r"""
        Returns a new tensor with its shape expanded.

        ``unsqueeze`` inserts a new axis of size one in the specified 
        ``axis``. For example a tensor of shape :math:`(N_1, N_2, N_3)` 
        with :math:`\text{axis}=(0, 2)` will output a tensor of shape 
        :math:`(1, N_1, 1, N_2, N_3)`.

        Note
        ----
        The returned tensor shares the storage with the input tensor, 
        so changing the contents of one will change the contents of the 
        other.

        See Also
        --------
        :meth:`giagrad.Tensor.squeeze`
        :func:`numpy.expand_dims`
        
        Parameters
        ----------
        axis: (int, ...) or int, optional
            Axis to be inserted with size one in the output tensor.

        Examples
        --------
        >>> t = Tensor.empty(2, 2, 2).uniform()                                                   
        >>> t
        tensor: [[[0.54203224 0.4911729 ]
                  [0.29304293 0.9672827 ]]
        ...
                 [[0.18163016 0.34806943]
                  [0.30323076 0.3647484 ]]]
        >>> t.unsqueeze(axis=(0,2))                                                               
        tensor: [[[[[0.54203224 0.4911729 ]
                    [0.29304293 0.9672827 ]]]
        ...
        ...
                  [[[0.18163016 0.34806943]
                    [0.30323076 0.3647484 ]]]]] fn: UnSqueeze(axis = (0, 2))
        >>> t.unsqueeze(axis=(0,2)).shape
        (1, 2, 1, 2, 2)
        """
        return Tensor.comm(sops.UnSqueeze(axis), self)

    # ***** statisitcs *****
    def var(self, axis=None, ddof=1, keepdims=False):
        r"""
        Calculates the variance over the axis specified by ``axis``.
        
        The variance (:math:`\sigma^2`) is calculated as:

        .. math::
            \sigma^2 = \frac{1}{N-\text{ddof}}\sum_{i=0}^{N-1}(x_i-\bar{x})^2

        If keepdims is True, the output tensor is of the same size as 
        the input except in the ``axis`` where it is of size 1. Otherwise, 
        every ``axis`` is squeezed, leading to an output tensor with 
        fewer dimensions. If no ``axis`` is supplied all data is reduced
        to a scalar value.

        Parameters
        ----------
        axis: (int, ...) or int or None, default: None
            The dimension or dimension to reduce. If None, var reduces 
            all dimensions.
        ddof: int, default: 1
            Degrees of freedom substracted to N. ``ddof=1`` equals sample
            variance, ``ddof=0`` equals population variance.
        keepdims: bool, default: False
            Whether te output tensor should retain the reduced dimensions.

        Examples
        --------
        >>> a = Tensor.empty(2, 2, 4, dtype=int).uniform(-10, 10)                                          
        >>> a
        tensor: [[[ 3  6  2 -5]
                  [ 7  3 -4 -8]]
        ...
                 [[ 1  2 -6  6]
                  [ 4  9  0 -5]]]
        >>> a.var()
        tensor: 24.808594 fn: Div
        >>> a.var((1, 2), ddof=1)                                                                         
        tensor: [30.         26.26785714] fn: Div
        >>> a.std((1, 2), ddof=1)**2
        tensor: [30.         26.26785714] fn: Pow
        """
        # numerator
        numer = (self - self.mean(axis=axis, keepdims=True))**2
        numer = numer.sum(axis, keepdims=keepdims)
        # denominator
        axis = (axis,) if isinstance(axis, int) else axis
        denom = self.size if axis is None else prod(self.shape[i] for i in axis) 
        denom -= ddof
        return numer / denom

    def std(self, axis=None, ddof=1, keepdims=False, eps=.0):
        r"""
        Calculates the standard deviation over the axis specified by ``axis``.
        
        The standard deviation (:math:`\sigma`) is calculated as:

        .. math::
            \sigma = \sqrt{\frac{1}{N-\text{ddof}}\sum_{i=0}^{N-1}(x_i-\bar{x})^2 + \epsilon}

        If keepdims is ``True``, the output tensor is of the same size as 
        input except in the ``axis`` where it is of size 1. Otherwise, 
        every ``axis`` is squeezed, leading to an output tensor with 
        fewer dimensions. If no ``axis`` is supplied all data is reduced
        to a scalar value. Optional parameter ``eps`` could be used for 
        numerical stability.

        Parameters
        ----------
        axis: (int, ...) or int or None, default: None
            The dimension or dimension to reduce. If None, std reduces 
            all dimensions.
        ddof: int, default: 1
            Degrees of freedom substracted to N. ``ddof=1`` equals sample
            variance, ``ddof=0`` equals population variance.
        keepdims: bool, default: False
            Whether te output tensor should retain the reduced dimensions.
        eps: float, default: 0.0
            Epsilon value added to :math:`\mathrm{Var}[x]` for numerical 
            stability.

        Examples
        --------
        >>> a = Tensor.empty(2, 2, 3, dtype=int).uniform(0, 10)                                            
        >>> a
        tensor: [[[2 8 5]
                  [8 4 6]]
        ...
                 [[7 9 6]
                  [3 3 6]]]
        >>> a.std()                                                                                       
        tensor: 2.1392496 fn: Pow
        >>> a.std((0, 1), keepdims=True, ddof=1)                                                          
        tensor: [[[2.94392029 2.94392029 0.5       ]]] fn: Pow
        >>> a.std((0, 1), keepdims=True, ddof=1).ndim 
        3
        >>> a.std((0, 1), keepdims=True, ddof=1, eps=.1)                                                
        tensor: [[[2.96085573 2.96085573 0.59160798]]] fn: Pow
        >>> (a.var((0, 1), keepdims=True, ddof=1) + .1).sqrt()                                             
        tensor: [[[2.96085573 2.96085573 0.59160798]]] fn: Pow
        """
        return (self.var(axis, ddof, keepdims) + eps).sqrt()


    # ***** other functions *****
    def einsum(
        self,
        subscripts, 
        *operands,
        optimize=False
    ) -> Tensor:
        """ 
        Computes Einstein summation convention on self and input operands.
    
        Adapted from :func:`numpy.einsum`.

        The Einsum function enables the computation of various 
        multi-dimensional linear algebraic array operations using a 
        shorthand notation based on the Einstein summation convention. 
        
        A non-exhaustive list of these operations, which can be computed 
        by `einsum`, is shown below along with examples:

        * Trace of a tensor.
        * Return a diagonal.
        * Array axis summations.
        * Transpositions and permutations.
        * Matrix multiplication and dot product.
        * Vector inner and outer products.
        * Broadcasting, element-wise and scalar multiplication.
        * Tensor contractions.

        The subscripts string is a comma-separated list of subscript labels
        (letters in [a-zA-Z]), where each label refers to a dimension of 
        the corresponding operand.
        Whenever a label is repeated it is summed, so ``a.einsum('i,i', b)``
        is equivalent to :py:func:`Tensor(np.inner(a.data, b.data), 
        requires_grad=True) <numpy.inner>`. If a label appears only once, 
        it is not summed, so ``a.einsum('i')`` produces a view of ``a`` 
        with no changes. A further example ``a.einsum('ij,jk', b)`` 
        describes traditional matrix multiplication and is equivalent to 
        :py:func:`a.matmul(b) <Tensor.matmul>`. Repeated subscript labels 
        in one operand take the diagonal. For example, ``a.einsum('ii')`` 
        is equivalent to :py:func:`Tensor(np.trace(a.data), requires_grad=True) 
        <numpy.trace>`.

        In *explicit mode* the output can be directly controlled by
        specifying output subscript labels.  This requires the
        identifier '->' as well as the list of output subscript labels.
        This feature increases the flexibility of the function since
        summing can be disabled or forced when required. The call
        ``a.einsum('i->')`` is like :py:func:`a.sum(axis=-1) <Tensor.sum>`,
        and ``a.einsum('ii->i')`` is like :py:func:`np.diag(a) <numpy.diag>`.
        The difference is that `einsum` does not allow broadcasting by default.
        Additionally ``a.einsum('ij,jh->ih', b)`` directly specifies the
        order of the output subscript labels and therefore returns matrix
        multiplication, unlike the example above in implicit mode.

        To enable and control broadcasting, use an ellipsis. Default
        NumPy-style broadcasting is done by adding an ellipsis to the 
        left of each term, like ``a.einsum('...ii->...i')``. To take the 
        trace along the first and last axes, you can do ``a.einsum('i...i')``, 
        or to do a matrix-matrix product with the left-most indices 
        instead of rightmost, one can do ``a.einsum('ij...,jk...->ik...', b)``.

        A few important notes: the equation may contain whitespaces 
        between different elements (subscripts, ellipsis, arrow, and comma), 
        but something like '...' is not valid. An empty string ('') is 
        valid for scalar operands.
        
        Warning
        --------
        *implicit mode* not supported.

        Parameters
        ----------
        subscripts: str
            Specifies the subscripts for summation.
        *operands: list of Tensor or array_like
            These are the tensors for the operation.
        optimization: {False, True, 'greedy', 'optimal'}, default: `False`
            Controls if intermediate optimization should occur. No 
            optimization will occur if False and True will default to 
            the ‘greedy’ algorithm.

        Examples
        --------
        >>> a = Tensor(np.arange(25).reshape(5,5))                                             
        >>> b = Tensor(np.arange(5))                                                           
        >>> c = Tensor(np.arange(6).reshape(2,3))

        Trace of a matrix:

        >>> a.einsum('ii')
        tensor: 60. fn: Einsum
        >>> np.trace(a.data)                                                                   
        60

        Extract the diagonal (requires explicit form):
        
        >>> a.einsum('ii->i')
        tensor: [ 0  6 12 18 24] fn: Einsum
        >>> np.diag(a.data)                                                                    
        array([ 0,  6, 12, 18, 24])

        Sum over an axis (requires explicit form):
        
        >>> a.einsum('ij->i')                                                                  
        tensor: [ 10  35  60  85 110] fn: Einsum
        >>> np.sum(a.data, axis=1)                                                             
        array([ 10,  35,  60,  85, 110])

        For higher dimensional arrays summing a single axis can be done 
        with ellipsis:
        
        >>> a.einsum('...j->...')
        tensor: [ 10  35  60  85 110] fn: Einsum

        Compute a matrix transpose, or reorder any number of axes:
        
        >>> c.einsum('ij->ji')
        tensor: [[0 3]
                 [1 4]
                 [2 5]] fn: Einsum
        >>> c.T
        tensor: [[0 3]
                 [1 4]
                 [2 5]] fn: Swapaxes(0, 1)

        Vector inner products:

        >>> b.einsum('i,i->', b)                                                                                     
        tensor: 30. fn: Einsum
        >>> np.inner(b.data, b.data)                                                                                 
        30

        Matrix vector multiplication:

        >>> a.einsum('ij,j->i', b)                                                                                   
        tensor: [ 30  80 130 180 230] fn: Einsum
        >>> a.einsum('...j,j->...', b)                                                                                
        tensor: [ 30  80 130 180 230] fn: Einsum
        >>> b @ a.T                                                                                                  
        tensor: [ 30  80 130 180 230] fn: Matmul

        Broadcasting and scalar multiplication:
        
        >>> c.einsum('...,...->...', 3)                                                                               
        tensor: [[ 0.  3.  6.]
                 [ 9. 12. 15.]] fn: Einsum
        >>> c.einsum('ij,->ij', 3)                                                                                    
        tensor: [[ 0.  3.  6.]
                 [ 9. 12. 15.]] fn: Einsum
        >>> c * 3
        tensor: [[ 0.  3.  6.]
                 [ 9. 12. 15.]] fn: Mul

        Outer product:
        
        >>> b.einsum('j,i->ij', np.arange(2)+1)                                                                      
        tensor: [[0 1 2 3 4]
                 [0 2 4 6 8]] fn: Einsum
        >>> np.outer(np.arange(2)+1, b.data)
        array([[0, 1, 2, 3, 4],
               [0, 2, 4, 6, 8]])

        Tensor contraction:
        
        >>> a = Tensor(np.arange(60.).reshape(3,4,5))                                                                
        >>> b = Tensor(np.arange(24.).reshape(4,3,2))                                                                
        >>> a.einsum('ijk,jil->kl', b)                                                                               
        tensor: [[4400. 4730.]
                 [4532. 4874.]
                 [4664. 5018.]
                 [4796. 5162.]
                 [4928. 5306.]] fn: Einsum
        >>> np.tensordot(a.data, b.data, axes=([1,0], [0,1]))                                                        
        array([[4400., 4730.],
               [4532., 4874.],
               [4664., 5018.],
               [4796., 5162.],
               [4928., 5306.]])
        """
        operands = (self,) + operands
        if "->" in subscripts:
            in_labels, out_labels = subscripts.replace(' ', '').split("->")
            implicit = False
        else:
            raise ValueError("Implicit mode not supported.")

        return Tensor.comm(
            oops.Einsum(in_labels, out_labels, optimize), 
            *operands
        )
