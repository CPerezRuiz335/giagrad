from giagrad.tensor import Tensor
import numpy as np
from typing import List, Tuple
from giagrad.optim.SGD import Optimizer

class Adam(Optimizer):
    r"""Implements Adam algorithm. 
    
    This algorithm calculates the exponential moving average of gradients and square gradients. 
    And the parameters of β1 and β2 are used to control the decay rates of these moving averages.
    
    Based on `PyTorch Adam`_.

    
    
    .. _PyTorch Adam: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam

    Attributes
    ----------
    params: iterable of Tensor
        Iterable of parameters to optimize.
    lr: float, default: 0.001
        Learning rate.
    betas: tuple(float,float), default: (0.9,0.999)
        Betas.
    eps: float, default: 1e-8
        Epsilon value.
    weight_decay: float, default: 0
        Weight decay (L2 penalty).
    maximize: bool, default: False
        Maximize the params based on the objective, instead of minimizing.
    amsgrad: bool, default: False
        Option to use the AMSGrad variant of this algorithm.

    .. _On the Convergence of Adam and Beyond: https://openreview.net/pdf?id=ryQu7f-RZ
    """

    def __init__(
            self, 
            params: List[Tensor],
            lr:float = 1e-3,
            betas:Tuple[float,float] = (0.9,0.999),
            eps:float = 1e-8,
            weight_decay:float = 0.,
            maximize:bool = False,
            amsgrad:bool = False
        ):
        super().__init__(params)
        self.lr, self.eps, self.weight_decay = lr, eps, weight_decay
        self.beta1, self.beta2 = betas
        self.maximize, self.amsgrad = maximize, amsgrad
        self.m     = [np.zeros(p.shape) for p in self.params]
        self.v     = [np.zeros(p.shape) for p in self.params]
        self.vmax_ = [np.zeros(p.shape) for p in self.params]

    #https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
    def step(self):
        for t, m, v, vmax_ in zip(self.params, self.m, self.v, self.vmax_):
            if self.maximize:
                g = -t.grad.copy()
            else:
                g = t.grad.copy()
            if self.weight_decay != 0:
                g += self.weight_decay * t.data
            m[:] = np.add(self.beta1 * m, (1-self.beta1) * g)
            v[:] = np.add(self.beta2 * v, (1-self.beta2) * g**2)
            m_ = m/(1-self.beta1**self.ite)
            v_ = v/(1-self.beta2**self.ite)
            if self.amsgrad:
                vmax_[:] = np.maximum(vmax_, v_)
                t.data -= self.lr * m_/(np.sqrt(vmax_+self.eps))
            else:
                t.data -= self.lr * m_/(np.sqrt(v_+self.eps))
        self.ite += 1