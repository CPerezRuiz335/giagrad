from typing import List, Any
from collections import OrderedDict
import numpy as np
from numpy.typing import NDArray
from giagrad.tensor import Tensor
from abc import ABC, abstractmethod

class Module(ABC):
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

    def parameters(self, _tmp=[], _head=True) -> List[Tensor]:
        # completely unnecesary but fancy
        fn = lambda x: x.parameters(_tmp=_tmp, _head=False) if isinstance(x, Module) else [x]
        for x in self.__odict__.values(): _tmp.extend(fn(x))
        return _tmp if _head else []

    def zero_grad(self):
        for t in self.parameters():
            t.grad = np.zeros_like(t.grad)

    @abstractmethod
    def __call__(self, x) -> Tensor:
        raise NotImplementedError(f"__call__ not implemented in {type(self)}")