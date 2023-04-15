from giagrad.tensor import Tensor
from giagrad.nn.module import Module
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import math

def _random_dims_to_zero(r: NDArray, p: float, first_n_axis: int):
    n = math.prod(r.shape[:first_n_axis]) 
    rng = np.random.default_rng()

    for _ in range(n):
        if rng.random() > p:
            idx = rng.integers(r.shape[:first_n_axis]) 
            r[tuple(idx)] *= 0

class DropoutNd(Module):
    r"""
    Randomly zeroes specified axes of the input tensor with probability :attr:`p`.

    During training, specified axes are zeroed with probability :math:`p`,
    and the remaining elements are scaled up by a factor of :math:`\frac{1}{1-p}` to 
    preserve the expected value of the output. During inference, the dropout layer 
    does not modify the input tensor.

    
    In a tensor of shape :math:`(N, C, H, W)`, 1D, 2D or even 3D slices can be zeroed,
    and this can be specified by paramter ``axis``. If no axes are supplied it will 
    zero out entire channels by default. 

    Inherits from: :class:`Module`.

    See Also
    --------
    :class:`Dropout`:
        For a more efficient way of zeroing only scalars in any dimension.

    :meth:`~Module.train`
    :meth:`~Module.eval`

    Attributes
    ----------
    p: float, default: 0.5
        Probability of axes to be zeroed.
    axis: int, optional
        Number of dimensions to be zeroed during training.

    Examples
    --------
    >>> a = Tensor.empty(2, 2, 2, 3).ones()
    >>> a
    tensor: [[[[1. 1. 1.]
               [1. 1. 1.]]
    ...
              [[1. 1. 1.]
               [1. 1. 1.]]]
    ...
    ...
             [[[1. 1. 1.]
               [1. 1. 1.]]
    ...
              [[1. 1. 1.]
               [1. 1. 1.]]]]

    Setting ``axis`` = 1 1D slices will be zeroed in each channel.
    
    >>> dropout = nn.DropoutNd(p=0.5, axis=1)
    >>> dropout(a)
    tensor: [[[[0. 0. 0.]
               [0. 0. 0.]]
    ...
              [[2. 2. 2.]
               [2. 2. 2.]]]
    ...
    ...
             [[[2. 2. 2.]
               [0. 0. 0.]]
    ...
              [[2. 2. 2.]
               [0. 0. 0.]]]] grad_fn: DropoutNd(p = 0.5, axis = 1)
    """
    def __init__(self, p: float, axis: Optional[int] = None):
        super().__init__()
        self.p = p
        self.__gain = 1 / (1 - self.p)
        self.__drop_axis = axis
        
    def __forward(self, x: Tensor) -> Tensor:
        if self.__drop_axis is None:
            self.__drop_axis = x.ndim - 2
        
        r = np.ones_like(x.data)
        _random_dims_to_zero(r, self.p, x.ndim - self.__drop_axis)
        return x * r * self.__gain
    
    def __check(self, ndim: int):
        if self.__drop_axis and self.__drop_axis >= ndim-1:
            raise ValueError("Does not make sense to dropout and entire observation\n" \
                             + f"dropout axes: {self.__drop_axis}, input: {ndim}")

    def __call__(self, x: Tensor) -> Tensor:
        self.__check(x.ndim)
        if self._train:
            out = self.__forward(x)
        return out

    def __str__(self):
        return f"DropoutNd(p = {self.p}, axis = {self.__drop_axis})"

class Dropout(DropoutNd):
    r"""
    Randomly sets some of the input tensor elements to zero during
    training using a Bernoulli distribution with a probability of :attr:`p`. 

    Each elements is independently zeroed out every time it is called. This 
    technique is effective for regularization and preventing the co-adaptation 
    of neurons, as explained in the paper titled  `Improving neural networks by 
    preventing co-adaptation of feature detectors`_. 

    Additionally, during training, the output is scaled by a factor of :math:`\frac{1}{1-p}`. 
    During evaluation, the module performs an identity function. 

    Inherits from: :class:`DropoutNd`.

    Attributes
    ----------
    p: float, default: 0.5
       Probability of each element to be zeroed.

    Examples
    --------
    >>> a = Tensor.empty(2, 3).ones()
    >>> dropout = nn.Dropout(p=0.5)
    >>> dropout(a)
    tensor: [[0. 2. 0.]
            [2. 0. 0.]] grad_fn: Dropout(p = 0.5)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """
    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def __check(self, ndim: int):
        # Dropout can set any scalar value to zero
        ...

    def __forward(self, x: Tensor) -> Tensor:
        r = np.random.binomial(1, 1-self.p, size=x.shape)
        return x * r * self.__gain

    def __str__(self):
        return f"Dropout(p = {self.p})"