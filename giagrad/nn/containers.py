from __future__ import annotations
from typing import List, Any, Callable, Optional, overload
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
        OrderedDict([('l1', Layer(in=784, out=100, bias=True)),
                     ('l2', Layer(in=100, out=10, bias=True))])

    .. note::
        For the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child if you want to have it 
        in ``__odict__``.

    :ivar training:  Represents whether this module is in training or evaluation mode.
    :vartype training: bool, default: True

    .. rubric:: Methods
    """
    def __init__(self):
        self.training = True

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

    def __getattr__(self, attr: Any):
        try:
            out_module = self.__odict__[attr]
        except AttributeError:
            raise AttributeError(f"{attr} is not a subModule")
        return out_module

    def add_module(self, module, name=None):
        r"""
        Adds a child module to the current module.

        The module can be accessed as an attribute using the given name. 

        Parameters
        ----------
        module: Module
            Child module to be added to the current module.
        name: str, optional
            Name of the child module. If no name is supplied its name becomes `module\%i` where
            \%i is the number of submodules already defined in ``self``.

        Examples
        --------
        >>> mod = nn.Sequential()
        >>> mod.add_module(nn.Linear(10, 10))
        >>> mod.module0
        Layer(in=10, out=10, bias=True)
        """
        if isinstance(module, Module):
            if name != None:
                self.__odict__[name] = module
            else:
                key = f"module{len(self.__odict__.keys())}"
                self.__odict__[key] = module

    def train(self):
        """
        Sets all submodules in training mode including self.

        See the documentation of the particularly modules you are using to 
        check whether they will be affected by training/evaluation mode, e.g.
        :class:`Dropout`, :class:`DropoutNd`, etc.
        """
        self.training = True 
        for subModule in self.__odict__.values():
            if isinstance(subModule, Module):
                subModule.train()

    def eval(self):
        """
        Sets all submodules in evaluation mode including self.

        See the documentation of the particularly modules you are using to 
        check whether they will be affected by training/evaluation mode, e.g.
        :class:`Dropout`, :class:`DropoutNd`, etc.
        """
        self.training = False
        for subModule in self.__odict__.values():
            if isinstance(subModule, Module):
                subModule.eval()

    def apply(self, fn: Callable):
        r"""Applies ``fn`` recursively to every submodule as well as self. 

        Typical use includes initializing the parameters of a model.

        Parameters
        ----------
        fn: callable, :class:`Module` -> None
            Function to be applied to each submodule, whether it is a Tensor or a Module
            ``fn`` must modify them in-place.

        Examples
        --------
        Using class Model defined in :class:`Module` example.

        >>> def init_weights(m):
        ...     if isinstance(m, nn.Linear):
        ...         m.w.ones()
        >>> m.apply(init_weights)
        >>> np.all(m.l1.w.data == 1) and np.all(m.l2.w.data == 1)  
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

                
class Sequential(Module):
    """TODO

    Inherits from: :class:`Module`.

    .. rubric:: Methods
    """
    @overload
    def __init__(self, *args: Module) -> None:
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]') -> None:
        ...
      
    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for name, module in args[0].items():
                self.add_module(module, name=name)
        else:
            for module in args:
                self.add_module(module)

    def append(self, module: Module) -> Sequential:
        r'''
        Calls ``self.add_module()`` function passing ``module`` as the one who's going to append.

        It's useful in many aspects. 
        
        '''
        self.add_module(module)
        return self
    
    def __call__(self, input: Tensor) -> Tensor:
        for module in self.__odict__.values():
            input = module(input)
        return input

    def __add__(self, other) -> Sequential:
        if isinstance(other, Sequential):
            res = Sequential()
            for mod in self:
                res.append(mod)
            for mod in other:
                res.append(mod)
            return res
        else:
            raise ValueError(f'Add operator supports only Sequential objects, but {str(type(other))} is given.')
    
    def __iadd__(self,other) -> Sequential:
        if isinstance(other, Sequential):
            for mod in other:
                self.add_module(mod)
            return self
        else:
            raise ValueError(f'Add operator supports only Sequential objects, but {str(type(other))} is given.')
