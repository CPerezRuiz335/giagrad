from giagrad.tensor import Tensor
import numpy as np
from typing import List, Tuple
from giagrad.optim.SGD import Optimizer

class Adamax(Optimizer):
    def __init__(
            self, 
            params: List[Tensor],
            lr:float = 1e-3,
            betas:Tuple[float,float] = (0.9,0.999),
            eps:float = 1e-8,
            weight_decay:float = 0.,
            maximize:bool = False
        ):
        super().__init__(params)
        self.lr, self.eps, self.weight_decay = lr, eps, weight_decay
        self.beta1, self.beta2 = betas
        self.maximize = maximize

    #https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
    def step(self):
        m, u = 0, 0
        for t in self.params:
            g = t.grad
            if self.weight_decay != 0:
                g += self.weight_decay * t.data
            m = self.beta1 * m + (1-self.beta1) * g
            u = max(self.beta2*u, np.abs(g+self.eps))
            t.data -= (self.lr * m)/((1-self.beta1**self.ite)*u)
            
        self.ite += 1