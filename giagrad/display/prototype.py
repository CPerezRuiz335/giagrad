from __future__ import annotations
from graphviz import Digraph # type: ignore
from giagrad.tensor import Tensor, Context
from typing import Callable, Dict
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

def tensor2node(tensor: Tensor, options: Dict, isop=False):

    if isop:
        return dict(
        label=str(tensor._ctx),
        fontsize=FONTSIZE
        )
    else:
        if options.get('shape'):
            label = f"{tensor.name} | {tensor.shape}"
        else:
            grad_txt = str(np.round(tensor.grad, decimals=1)).replace('\n', '\\n').replace('. ', ' ').replace('.]' , ']')
            data_txt = str(np.round(tensor.data, decimals=1)).replace('\n', '\\n').replace('. ', ' ').replace('.]' , ']')
            if options.get('grad'):
                label = f"{tensor.name} | {{{data_txt} | grad:\\n{grad_txt}}}"
            else:
                label = f"{tensor.name} | {{{data_txt}}}"
        
        return dict( 
        label=label,
        shape="record",
        fontsize=FONTSIZE
        )

def draw_dot(root, rankdir='LR', **options):
    """
    Returns autograd computational graph as a pygraphviz.Digraph object.
    
    Every node is either an operator or a tensor, when ``shape``
    is False ``data`` is not displayed, ``grad`` can be displayed if it is specified. 
    For backward pass visualization ``retain_graph`` must be enabled, 
    otherwise the computational graph has been deleted, see :meth:`~giagrad.Tensor.backward`.
    
    Parameters
    ----------
    rankdir: str, default: 'LR'
        Direction to draw the graph, see `Graphviz rankdir`_.

    Keyword Arguments
    -----------------
    shape: bool, default: False
        Display the shape of the tensor and its name.
    grad: bool, default: False
        Display gradient when ``shape`` is False.

    Examples
    --------
    >>> a = Tensor([[-1.,-2.],[3.,-4.],[-5.,6.]], 
    ...           requires_grad=True, name='a')
    >>> b = a.mean()
    >>> b.name = 'b'
    >>> a2 = a.exp()
    >>> a2.name = "a'"
    >>> c = Tensor([[2.,2.],[2.,2.],[3.,3.]], 
    ...           requires_grad=True, name='c')
    >>> c = c * a2.log().relu()
    >>> c.name = "c'"
    >>> d = b * c
    >>> d.name = 'd'
    >>> e = Tensor([[5.,5.,5.]],
    ...           requires_grad=True, name='e')
    >>> f = e @ d
    >>> f.name = 'f'
    >>> g = f.sum()
    >>> g.name = 'g'
    >>> draw_dot(g, rankdir='TB')

    .. _Graphviz rankdir: https://graphviz.org/docs/attr-types/rankdir/ 
    """
    dot = Digraph(strict=True, graph_attr={'rankdir': rankdir})

    def _draw(parent: Tensor, tensor: Tensor):
        tensor_name = tensor.name if tensor.name else f"{tensor.name} {hash(tensor)}"
        parent_name = parent.name if parent.name else f"{parent.name} {hash(parent)}"
        
        # Add tensor node
        dot.node(name=tensor_name, **tensor2node(tensor, options))
        # Add context operator node
        dot.node(name=f"{tensor_name} {tensor._ctx}", **tensor2node(tensor, options, isop=True, ))
        # Add parent node
        dot.node(name=parent_name, **tensor2node(parent, options))
        # Add edges
        dot.edge(f"{tensor_name} {tensor._ctx}", tensor_name)
        dot.edge(parent_name, f"{tensor_name} {tensor._ctx}")

    trace(root, _draw)
    return dot