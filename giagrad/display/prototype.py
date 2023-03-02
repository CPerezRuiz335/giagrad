from __future__ import annotations
from graphviz import Digraph # type: ignore
from giagrad.tensor import Tensor, Context
from typing import Callable
import numpy as np

FONTSIZE = "12"

def trace(root: Tensor, fun: Callable):
    nodes = set()

    def build(tensor: Tensor):
        nodes.add(tensor)
        if (context := tensor._ctx):
            for p in context.parents:
                fun(p, tensor)
                if p not in nodes:
                    build(p)

    build(root)

def tensor2node(tensor: Tensor, isop=False):

    if isop:
        return dict(
        label=str(tensor._ctx),
        fontsize=FONTSIZE
        )
    else:
        grad_txt = str(np.round(tensor.grad, decimals=1)).replace('\n', '\\n').replace('. ', ' ').replace('.]' , ']')
        data_txt = str(np.round(tensor.data, decimals=1)).replace('\n', '\\n').replace('. ', ' ').replace('.]' , ']')

        return dict( 
        label=f"{tensor.name} | {{data:\\n{data_txt} | grad:\\n{grad_txt}}}",
        shape="record",
        fontsize=FONTSIZE
        )

def draw_dot(root, format_='svg', rankdir='LR'):
    dot = Digraph(format=format_, strict=True, graph_attr={'rankdir': rankdir})
    

    def _draw(parent: Tensor, tensor: Tensor):
        tensor_name = tensor.name if tensor.name else f"{tensor.name} {hash(tensor)}"
        parent_name = parent.name if parent.name else f"{parent.name} {hash(parent)}"
        
        # Add tensor node
        dot.node(name=tensor_name, **tensor2node(tensor))
        # Add context operator node
        dot.node(name=f"{tensor_name} {tensor._ctx}", **tensor2node(tensor, isop=True))
        # Add parent node
        dot.node(name=parent_name, **tensor2node(parent))
        # Add edges
        dot.edge(f"{tensor_name} {tensor._ctx}", tensor_name)
        dot.edge(parent_name, f"{tensor_name} {tensor._ctx}")

    trace(root, _draw)
    return dot