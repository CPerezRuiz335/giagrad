from typing import List, Any, Callable
from collections import OrderedDict
import numpy as np
from numpy.typing import NDArray
from giagrad.tensor import Tensor
from abc import ABC, abstractmethod

class Module(ABC):
    def __init__(self):
        self._train: bool = True

    def __new__(cls, *args, **kwargs):
        instance = object.__new__(cls)
        instance.__odict__ = OrderedDict()
        return instance 

    def __setattr__(self, key, value):
        # This guarantees that __odict__ only saves instances 
        # of Tensor or Module => useful for parameters method
        if key != '__odict__': 
            if isinstance(value, Tensor) or isinstance(value, Module):
                self.__odict__[key] = value
        object.__setattr__(self, key, value)

    def train(self):
        self._train = True 
        for subModule in self.__odict__.values():
            if isinstance(subModule, Module):
                subModule.train()

    def eval(self):
        self._train = False
        for subModule in self.__odict__.values():
            if isinstance(subModule, Module):
                subModule.eval()

    def apply(self, fn: Callable):
        for i, x in self.__odict__.items():
            self.__odict__[i] = fn(x)

    def parameters(self) -> List[Tensor]:
        out = []
        for x in self.__odict__.values(): 
            if isinstance(x, Tensor): out.append(x) 
            elif isinstance(x, Module): out.extend(x.parameters())
        return out

    def zero_grad(self):
        for t in self.parameters():
            t.grad = np.zeros_like(t.grad)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{type(self).__name__}\n\t" \
                + '\n\t'.join([str(m) for m in self.__odict__.values() if isinstance(m, Module)])

    @abstractmethod
    def __call__(self, x) -> Tensor:
        raise NotImplementedError(f"__call__ not implemented in {type(self)}")