from __future__ import annotations
from graphviz import Digraph
from giagrad.tensor import Tensor, Context
from typing import Callable

def trace(root: Tensor, fun: Callable):
    nodes = set()

    def build(tensor: Tensor):
        nodes.add(tensor)
        if (context := tensor._ctx):
            for p in context.parents:
                if isinstance(p, Tensor):
                    fun(p, tensor)
                    if p not in nodes:
                        build(p)

    build(root)

def draw_dot(root, format_='svg', rankdir='LR'):
    dot = Digraph(format=format_, strict=True, graph_attr={'rankdir': rankdir})

    def _draw(parent: Tensor, tensor: Tensor):
        parent_shape = 'x'.join(str(i) for i in parent.shape)
        tensor_shape = 'x'.join(str(i) for i in tensor.shape)
        if not parent.shape:
            parent_shape = f"data: {parent.data}"
        if not tensor.shape:
            tensor_shape = f"data: {tensor.data}"

        # Add tensor node
        dot.node(
            name=tensor.name, 
            label=f"{tensor.name} | {tensor_shape}",
            shape='record'
        )
        # Add context operator node
        dot.node(
            name=f"{tensor.name} {tensor._ctx}", 
            label=str(tensor._ctx)
        )
        # Add parent node
        dot.node(
            name=parent.name, 
            label=f"{parent.name} | {parent_shape}",
            shape='record'
        )

        # Add edges
        dot.edge(f"{tensor.name} {tensor._ctx}", tensor.name)
        dot.edge(parent.name, f"{tensor.name} {tensor._ctx}")


    trace(root, _draw)
    return dot