from typing import List 
from abc import ABC, abstractmethod

from giagrad.tensor import Tensor

class Optimizer(ABC):
    def __init__(self, params: List[Tensor]):
        self.params = [t for t in params if t.requires_grad]
        self.ite = 1

    @abstractmethod
    def step(self):
        """Performs a single optimization step/epoch (parameter update)."""
        raise NotImplementedError(f"step not implemented for {type(self)}")

    def zero_grad(self):
        """Sets the gradients of all optimized tensors to zero."""
        for p in self.params: 
            p.grad *= 0 # np.zeros_like(p.grad, dtype=np.float32) 