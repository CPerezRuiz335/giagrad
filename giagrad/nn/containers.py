from __future__ import annotations
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Any, Callable, Optional, overload, Iterator, Tuple

import numpy as np
from numpy.typing import NDArray

from giagrad.tensor import Tensor

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
        print(attr)
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
            Function to be applied to each submodule, whether it is a 
            Tensor or a Module ``fn`` must modify them in-place.

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
        return (
            f"{type(self).__name__}\n\t" 
            + '\n\t'.join([
                f"{name}: {mod}" 
                for name, mod 
                in self.__odict__.items() 
                if isinstance(mod, Module)
            ])
        )

                
class Sequential(Module):
    """
    A sequential container. 

    Based on PyTorch's `Sequential`_ documentation.

    Modules will be added to it in the order they are passed in the 
    constructor. Alternatively, an OrderedDict of modules can be passed 
    in. The forward() method of Sequential accepts any input and 
    forwards it to the first module it contains. It then “chains” 
    outputs to inputs sequentially for each subsequent module, finally 
    returning the output of the last module.

    Example:

    .. code-block:: python

        # Using Sequential to create a small model. When `model` is run,
        # input will first be passed to `Conv2D(1,20,5)`. The output of
        # `Conv2D(1,20,5)` will be used as the input to the first
        # `ReLU`; the output of the first `ReLU` will become the input
        # for `Conv2D(20,64,5)`. Finally, the output of
        # `Conv2D(20,64,5)` will be used as input to the second `ReLU`
        model = nn.Sequential(
                  nn.Conv2D(1,20,5),
                  nn.ReLU(),
                  nn.Conv2D(20,64,5),
                  nn.ReLU()
                )

        # Using Sequential with OrderedDict. This is functionally the
        # same as the above code
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2D(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2D(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    

    Inherits from: :class:`Module`.

    Examples
    --------
    Sequentials can be added together or in-place, and iterated too.

    >>> model = nn.Sequential(
                nn.Linear(500),
                nn.ReLU(),
                nn.Linear(10)
            )
    >>> model += model
    >>> for key, subModule in model:
    ...     print(key, subModule)
    module0 Layer(out_features=500, bias=True)
    module1 ReLU
    module2 Layer(out_features=10, bias=True)
    module3 Layer(out_features=500, bias=True)
    module4 ReLU
    module5 Layer(out_features=10, bias=True)
    

    .. rubric:: Methods

    .. _Sequential: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
    """
    @overload
    def __init__(self, *args: Module) -> None:
        ...

    @overload
    def __init__(self, arg: OrderedDict[str, Module]) -> None:
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
        Appends a new module to self.

        See Also
        --------
        :meth:`giagrad.nn.containers.Module.add_module`

        Examples
        --------
        >>> s = nn.Sequential(nn.Linear(10), nn.Dropout(p=.2))
        >>> for module in s:
        ...     print(module)
        Layer(out_features=10, bias=True)
        Dropout(p=0.2)
        '''
        self.add_module(module)
        return self
    
    def __call__(self, input: Tensor) -> Tensor:
        for module in self.__odict__.values():
            input = module(input)
        return input

    def __iter__(self) -> Iterator[Tuple[str, Module]]:
        for key, subModule in self.__odict__.copy().items():
            yield key, subModule

    def __add__(self, other: Sequential) -> Sequential:
        if isinstance(other, Sequential):
            res = Sequential()
            for _, mod in self:
                res.append(mod)
            for _, mod in other:
                res.append(mod)
            return res
        else:
            raise ValueError(
                'Add operator only supports Sequential objects, ' 
                f'but {type(other)} is given.'
            )
    
    def __iadd__(self, other: Sequential) -> Sequential:
        if isinstance(other, Sequential):
            for _, mod in other:
                self.add_module(mod)
            return self
        else:
            raise ValueError(
                'In-place add operator only supports Sequential objects, '
                f'but {type(other)} is given.'
            )
