from __future__ import annotations
from graphviz import Digraph
from giagrad.tensor import Tensor, Context

def trace(root: Tensor):
    nodes, edges = set(), set()
    def build(tensor: Tensor):
        if isinstance(tensor, Tensor) and tensor not in nodes:
            nodes.add(tensor)
            print(tensor.name)
            if (context := tensor._ctx):
                for p in context.parents:
                    try:
                        edges.add((p, tensor))
                    except TypeError:
                        continue
                    build(p)

    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=n.name, label=f"{n.name}", shape='record')
        if n._ctx is not None:
            print(n.name + str(n._ctx))
            dot.node(name=n.name + str(n._ctx), label=str(n._ctx))
            dot.edge(n.name + str(n._ctx), n.name)
    
    for n1, n2 in edges:
        dot.edge(n1.name, n2.name + str(n2._ctx))
    
    return dot
