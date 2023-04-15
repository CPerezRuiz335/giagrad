from typing import List, Any, Callable
from collections import OrderedDict
import numpy as np
from numpy.typing import NDArray
from giagrad.tensor import Tensor
from abc import ABC, abstractmethod

class Module(ABC):
    r"""
    Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes:
    
    .. code-block:: python

        import giagrad.nn as nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(28*28, 100)
                self.l2 = nn.Linear(100, 10)

            def __call__(self, x):
                x = self.l1(x).relu()
                return self.l2(x).relu()

    Submodules assigned in this way will be registered thanks to ``Model``
    constructor in ``__odict__`` variable in the same order as they are defined.
    
        >>> m = Model()
        >>> m.__odict__
        OrderedDict([('l1', Layer(in = 784, out = 100, bias = True)),
                     ('l2', Layer(in = 100, out = 10, bias = True))])

    .. note::
        For the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child if you want to have it 
        in ``__odict__``.

    :ivar _train:  Represents whether this module is in training or evaluation mode.
    :vartype _train: bool, default: True

    .. rubric:: Methods
    """
    def __init__(self):
        self._train = True

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
        """
        Sets all submodules, including self, in training mode.

        See the documentation of the particularly modules you are using to 
        check whether they will be affected by training/evaluation mode, e.g.
        :class:`Dropout`, :class:`DropoutNd`, etc.
        """
        self._train = True 
        for subModule in self.__odict__.values():
            if isinstance(subModule, Module):
                subModule.train()

    def eval(self):
        """
        Sets all submodules, including self, in evaluation mode.

        See the documentation of the particularly modules you are using to 
        check whether they will be affected by training/evaluation mode, e.g.
        :class:`Dropout`, :class:`DropoutNd`, etc.
        """
        self._train = False
        for subModule in self.__odict__.values():
            if isinstance(subModule, Module):
                subModule.eval()

    def apply(self, fn: Callable):
        r"""Applies ``fn`` recursively to every submodule as well as self. 

        Typical use includes initializing the parameters of a model.

        Parameters
        ----------
        fn: callable, :class:`Module` -> None
            Function to be applied to each submodule, whether it is Tensor or Module
            must modify submodules in-place.

        Examples
        --------
        Using class Model defined in :class:`Module` example.

        >>> def init_weights(m):
        ...     if isinstance(m, nn.Linear):
        ...         m.w.ones()
        >>> m.apply(init_weights)
        >>> np.all(m.l1.w == 1) and np.all(m.l2.w == 1)  
        True
        """
        for x in self.__odict__.values():
            fn(x)
            if isinstance(x, Module):
                x.apply(fn)

    def parameters(self) -> List[Tensor]:
        """
        Returns an iterator over all submodules and self parameters.

        This is typically passed to an optimizer.

        Examples
        --------
        Using class Model defined in :class:`Module` example.

        >>> for p in model.parameters():
        ...     print(type(p), p.shape)
        <class 'giagrad.tensor.Tensor'> (100, 784)
        <class 'giagrad.tensor.Tensor'> (100,)
        <class 'giagrad.tensor.Tensor'> (10, 100)
        <class 'giagrad.tensor.Tensor'> (10,)
        """
        out = []
        for x in self.__odict__.values(): 
            if isinstance(x, Tensor): out.append(x) 
            elif isinstance(x, Module): out.extend(x.parameters())
        return out

    def zero_grad(self):
        """
        Sets gradients of all tensors returned by ``parameters`` to zero.
        """
        for t in self.parameters():
            t.grad *= 0

    def __repr__(self):
        return self.__str__()

    @abstractmethod 
    def __call__(self, *args, **kwargs) -> Tensor:
        """The forward method of every module must be the method ``__call__``."""
        raise NotImplementedError(f"__call__ not implemented for class {type(self)}")

    def __str__(self):
        return f"{type(self).__name__}\n\t" \
                + '\n\t'.join([str(m) for m in self.__odict__.values() if isinstance(m, Module)])