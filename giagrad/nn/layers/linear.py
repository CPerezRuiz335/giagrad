from giagrad.tensor import Tensor, Function
from giagrad.nn.containers import Module
from math import sqrt
from typing import Optional
import numpy as np

def overload(fun):
    def wrapper(self, *args, **kwargs):
        if len(args) == 1:
            return fun(self, in_features=None, out_features=args[0])
        if len(kwargs) == 1:
            # assume kwargs has out_features
            return fun(self, in_features=None, **kwargs)
        return fun(self, *args, **kwargs)
    return wrapper

class Linear(Module):
    r"""
    Densely-connected Neural Network layer: :math:`y = xA^T + b`.
    
    Both ``w`` and ``b`` are initialized from :math:`\mathcal{U}(\sqrt{-k}, \sqrt{k})`,
    where :math:`k = \frac{1}{\text{in_features}}`. 

    Inherits from: :class:`Module`.

    Attributes
    ----------
    w: Tensor
        Learnable weights of the layer of shape 
        :math:`(\text{out_features}, \text{in_features})`.
    b: Tensor, optional
        Learnable bias of the layer. Only exists when bias is True.

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    out_features: int, optinal
        Size of each output sample.
    bias: bool, default: True
        If set to False, the layer will not learn an additive bias. 
    
    Shape
    -----
    Input: 
        :math:`(*, H_{in})` where :math:`*` means any number of
        dimensions including none and :math:`H_{in} = \text{in_features}`.
    Output: 
        :math:`(*, H_{out})` where all but the last dimension
        are the same shape as the input and 
        :math:`H_{out} = \text{out_features}`.

    Examples
    --------
    >>> layer = nn.Linear(10, 5)
    >>> x = Tensor.empty(2, 10).uniform()
    >>> y = layer(x)
    >>> y.shape
    (2, 5)
    """ 
    @overload
    def __init__(
        self, 
        in_features: int, 
        out_features: int,  
        bias: bool = True
    ):
        super().__init__()
        self.w: Optional[Tensor] = None # uninitialized
        self.b: Optional[Tensor] = None # uninitialized

        self.bias = bias
        self.out_features = out_features
        self.in_features = in_features

    def __init_tensors(self, batch: int, in_features: int):
        if self.in_features is None:
            self.in_features = in_features  

        k = 1 / sqrt(in_features)
        self.w = Tensor.empty(self.out_features, in_features, requires_grad=True)
        self.w.uniform(a=-k, b=k)
        
        if self.bias:
            b = np.random.uniform(-k, k, size=(1, self.out_features))
            b = np.repeat(b, batch, axis=0)
            self.b = Tensor(b, requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        if self.w is None:
            self.__init_tensors(*x.shape[-2:])

        if self.bias: 
            return x @ self.w.T + self.b # x.gemm(alpha=1., b=self.w, c=self.b, trans_b=True)
        else: 
            return x @ self.w.T # x.gemm(alpha=1., b=self.w, trans_b=True)   

    def __str__(self):
        return (
            "Layer("
            + (f"in_features={self.in_features}, " if self.in_features else '')
            +  f"out_features={self.out_features}, "
            +  f"bias={self.bias})"
        )

