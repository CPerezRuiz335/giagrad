from __future__ import annotations
import giagrad.mlops as mlops
from giagrad.tensor import Tensor
from abc import ABC, abstractmethod
from giagrad.nn.containers import Module

class ReLU(Module):
    """
    Applies the Rectified Linear Unit (ReLU) function element-wise. 

    See Also
    --------
    :func:`giagrad.Tensor.relu`.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, t1: Tensor) -> Tensor:
        return Tensor.comm(mlops.ReLU(), t1)

    def __str__(self): 
        return "ReLU"

class ReLU6(Module):
    """
    Applies a modified version of ReLU with maximum size of 6.

    See Also
    --------
    :func:`giagrad.Tensor.relu6`.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, t1: Tensor) -> Tensor:
        return Tensor.comm(mlops.ReLU6(), t1)

    def __str__(self): 
        return "ReLU6"

class Hardswish(Module):
    """
    Applies the hardswish function, element-wise.

    See Also
    --------
    :func:`giagrad.Tensor.hardswish`.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, t1: Tensor) -> Tensor:
        return Tensor.comm(mlops.Hardswish(), t1)

    def __str__(self): 
        return "Hardswish"

class Sigmoid(Module):
    r"""
    Applies the element-wise function :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`

    See Also
    --------
    :func:`giagrad.Tensor.sigmoid`.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, t1: Tensor) -> Tensor:
        return Tensor.comm(mlops.Sigmoid(), t1)

    def __str__(self): 
        return "Sigmoid"

class ELU(Module):
    r"""
    Applies the Exponential Linear Unit (ELU) function element-wise.

    See Also
    --------
    :func:`giagrad.Tensor.elu`.

    Attributes
    ----------
    alpha: float, default: 1.0
        The :math:`\alpha` value for the ELU formulation.
    """
    def __init__(self, alpha: float = 1.):
        super().__init__()
        self.alpha = alpha

    def __call__(self, t1: Tensor) -> Tensor:
        return Tensor.comm(mlops.ELU(alpha=self.alpha), t1)

    def __str__(self): 
        return f"ELU(alpha={self.alpha})"

class SiLU(Module):
    """
    Applies the Sigmoid Linear Unit (SiLU) function, element-wise.

    See Also
    --------
    :func:`giagrad.Tensor.silu`.

    Attributes
    ----------
    beta: float
        Hyperparameter for SiLU formulation.
    """
    def __init__(self, beta: float = 1.):
        super().__init__()
        self.beta = beta

    def __call__(self, t1: Tensor) -> Tensor:
        return Tensor.comm(mlops.SiLU(beta=self.beta), t1)

    def __str__(self): 
        return f"SiLU(beta={self.beta})"

class Tanh(Module):
    """
    Applies the hyperbolic tangent function element-wise.

    See Also
    --------
    :func:`giagrad.Tensor.tanh`.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, t1: Tensor) -> Tensor:
        return Tensor.comm(mlops.Tanh(), t1)

    def __str__(self): 
        return f"Tanh"

class LeakyReLU(Module):
    r"""
    Applies element-wise 
    :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative_slope} * \min(0, x)`.

    See Also
    --------
    :func:`giagrad.Tensor.leakyrelu`.

    Attributes
    ----------
    neg_slope: float
        Controls de angle of the negative slope (which only affects negative input values). 
    """
    def __init__(self, neg_slope: float = 0.01):
        super().__init__()
        self.neg_slope = neg_slope

    def __call__(self, t1: Tensor) -> Tensor:
        return Tensor.comm(mlops.LeakyReLU(neg_slope=self.neg_slope), t1)

    def __str__(self): 
        return f"LeakyReLU(neg_slope={self.neg_slope})"

class SoftPlus(Module):
    r"""
    Applies element-wise 
    :math:`\text{SoftPlus}(x) = \frac{1}{\text{beta}} \cdot \log(1 + \exp(\text{beta} \times data_i))`.
    
    For numerical stability the implementation reverts to the linear 
    function when :math:`data_i \times \text{beta} > \text{limit}`.

    See Also
    --------
    :func:`giagrad.Tensor.softplus`.

    Attributes
    ----------
    beta: float
        The :math:`\beta` value for the Softplus formulation.
    limit: float
        Data times beta above this reverts to a linear function.
    """ 
    def __init__(self, beta: float = 1., limit: float = 20):
        super().__init__()
        self.beta = beta
        self.limit = limit

    def __call__(self, t1: Tensor) -> Tensor:
        return Tensor.comm(mlops.Softplus(limit=self.limit, beta=self.beta), t1)

    def __str__(self): 
        return f"SoftPlus(beta={self.beta}, lim={self.limit})"

class Mish(Module):
    r"""
    Applies Mish activation function element-wise.

    See Also
    --------
    :func:`giagrad.Tensor.mish`.

    Attributes
    ----------
    beta: float
        The :math:`\beta` value for the Softplus formulation.
    limit: float
        Data times beta above limit reverts to a linear function in 
        Softplus formulation.
    """
    def __init__(self, beta: float = 1., limit: float = 20):
        super().__init__()
        self.beta = beta
        self.limit = limit

    def __call__(self, t1: Tensor) -> Tensor:
        return Tensor.comm(mlops.Mish(limit=self.limit, beta=self.beta), t1)

    def __str__(self): 
        return f"Mish(beta={self.beta}, lim={self.limit})"

class GELU(Module):
    r"""
    Applies element-wise the function
    :math:`\text{GELU}(x) = x \times \Phi(x)`.

    See Also
    --------
    :func:`giagrad.Tensor.gelu`.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, t1: Tensor) -> Tensor:
        return Tensor.comm(mlops.GELU(), t1)

    def __str__(self): 
        return "GELU"

class QuickGELU(Module):
    r"""
    Applies :math:`\text{GELU}(x) = x \times \Phi(x)` activation 
    function with SiLU approximation.

    See Also
    --------
    :func:`giagrad.Tensor.quick_gelu`.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, t1: Tensor) -> Tensor:
        return Tensor.comm(mlops.SiLU(beta=1.702), t1)

    def __str__(self): 
        return "QuickGELU"

class Softmax(Module):
    """
    Applies the softmax function through 1-D slices specified by ``axis``.

    See Also
    --------
    :func:`giagrad.Tensor.softmax`.

    Attributes
    ----------
    axis: int
        The dimension along which Softmax will be computed (so every 
        slice along axis will sum to 1).

    """
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def __call__(self, t1: Tensor) -> Tensor:
        return Tensor.comm(mlops.Softmax(axis=self.axis), t1)

    def __str__(self): 
        return f"Softmax(axis={self.axis})"

class LogSoftmax(Module):
    """
    Applies the logsoftmax function through 1-D slices specified by ``axis``.

    See Also
    --------
    :func:`giagrad.Tensor.log_softmax`.

    Attributes
    ----------
    axis: int
        The dimension along which Softmax will be computed (so every 
        slice along axis will sum to 1).
    """
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def __call__(self, t1: Tensor) -> Tensor:
        return Tensor.comm(mlops.LogSoftmax(axis=self.axis), t1)

    def __str__(self): 
        return f"LogSoftmax(axis={self.axis})"