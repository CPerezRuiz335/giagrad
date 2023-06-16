from giagrad.tensor import Tensor
from giagrad.nn.containers import Module
from math import sqrt
from typing import Optional

class Linear(Module):
    r"""
    Densely-connected Neural Network layer: :math:`y = xA^T + b`.
    
    Both ``w`` and ``b`` are initialized from :math:`\mathcal{U}(\sqrt{-k}, \sqrt{k})`,
    where :math:`k = \frac{1}{\text{in_features}}`. 

    Inherits from: :class:`Module`.

    Attributes
    ----------
    w: Tensor
        Learnable weights of the layer of shape :math:`(\text{out_features}, \text{in_features})`.
    b: Tensor, optional
        Learnable bias of the layer. Only exists when bias is True.

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    out_features: int
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
        are the same shape as the input and :math:`H_{out} = \text{out_features}`.

    Examples
    --------
    >>> layer = nn.Linear(10, 5)
    >>> x = Tensor.empty(2, 10).uniform()
    >>> y = layer(x)
    >>> y.shape
    (2, 5)
    """ 
    def __init__(
        self, 
        out_features: int,  
        in_features: Optional[int] = None, 
        bias: bool = True
    ):
        super().__init__()
        self.bias = bias
        self.__out_features = out_features
        self.__in_features = in_features
        if in_features is not None:
            self.__init_tensors(in_features)

    def __init_tensors(self, in_features: int):
        k = 1 / sqrt(in_features)
        self.w = Tensor.empty(
            self.__out_features, in_features, 
            requires_grad=True
        ).uniform(a=-k, b=k)
        
        if self.bias:
            self.b = Tensor.empty(
                self.__out_features, 
                requires_grad=True
            ).uniform(a=-k, b=k)
        else:
            self.b = None

    def __call__(self, x: Tensor) -> Tensor:
        if self.__in_features is None:
            self.__init_tensors(in_features=x.shape[0])

        if self.bias: 
            return x @ self.w.T + self.b
        else: 
            return x @ self.w.T        

    def __str__(self):
        out, in_ = self.w.shape
        return f"Layer(in={in_}, out={out}, bias={self.bias})"

