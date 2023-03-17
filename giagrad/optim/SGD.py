from giagrad.tensor import Tensor
import numpy as np
from typing import List 
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, params: List[Tensor]):
        for p in params: p.requires_grad_()
        self.params = params
        self.ite = 1

    @abstractmethod
    def step(self):
        raise NotImplementedError(f"step not implemented for {type(self)}")

    def zero_grad(self):
        for p in self.params: 
            p.grad = np.zeros_like(p, dtype=np.float32) 


class SGD(Optimizer):
    def __init__(
            self, 
            params: List[Tensor], 
            lr: float = 0.001, 
            momentum: float = 0,
            weight_decay: float = 0,
            dampening: float = 0, 
            nesterov: bool = False, 
            maximize: bool = False
        ):
        assert not nesterov or (momentum != 0  and dampening == 0), \
            "Nesterov momentum requires a momentum and zero dampening"
        super().__init__(params)
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
        self.dampening, self.nesterov, self.maximize = dampening, nesterov, maximize
        
        self.b = [Tensor.zeros(*t.shape) for t in self.params] if self.momentum else []

  # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    def step(self):
        for t, b in zip(self.params, self.b):
            g = t.grad

            if self.weight_decay != 0:
                g += self.weight_decay * t.data

            if self.momentum != 0:
                if self.ite > 1:
                    b = self.momentum * b + (1 - self.dampening) * g
                else:
                    b = g 

                if self.nesterov:
                    g += self.momentum * b 
                else:
                    g = b 

            if self.maximize:
                t.data += self.lr * g 
            else:
                    t.data -= self.lr * g

        self.ite += 1