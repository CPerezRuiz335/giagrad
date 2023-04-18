from __future__ import annotations
from giagrad.tensor import Context, Tensor
from giagrad.nn import Module

def Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        for module in args:
            self.add_module(module)

    def append(self, module: Module) -> 'Sequential':
        self.add_module(module)
        return self
    
    def __call__(self, input):
        for module in self:
            if isinstance(module, Context):
                input = Tensor.comm(module, input, axis=1)
            elif isinstance(module, Module):
                input = module(input)
        return input