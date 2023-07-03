# https://github.com/rsokl/MyGrad/blob/master/src/mygrad/linalg/ops.py
from itertools import chain
from collections import Counter
from functools import reduce
from typing import Literal, Tuple, Union, List, Optional

import numpy as np
from numpy.typing import NDArray
from numpy.lib.stride_tricks import as_strided
from numpy.core.einsumfunc import _parse_einsum_input

from giagrad.tensor import Tensor, Function
from giagrad.mathops import collapse

Optimize = Literal[False, True, 'greedy', 'optimal']

def unique_from_end(in_str):
    """Return a string with all redundant characters removed,
    removing left-most redundant entries, i.e. "ijikik" -> "jik".
    """
    return reduce(
        lambda acc, x: acc+x if x not in acc else acc, 
        in_str[::-1], 
        ""
    )[::-1]


def merge_max_mappings(*mappings):
    """Merge dictionaries based on largest values in key->value.
    merge_max_mappings({"a":1, "b":4}, {"a":2}) == {"a":2, "b":4}
    """
    def merge_max(d1, d2):
        d1.update((k, v) for k, v in d2.items() if d1.get(k, 0) < v)
        return d1

    return reduce(merge_max, mappings, {})


def get_indices(item, seq):
    """Return the indices where `item` occurs in `seq`"""
    return (n for n, x in enumerate(seq) if x == item)


class Einsum(Function):
    def __init__(self, in_labels: str, out_labels: str, optimize: Optimize):
        super().__init__()
        self.in_labels = in_labels.split(',')
        self.out_labels = out_labels
        self.optimize = optimize    

        # cache counts the number of redundant tensor-label pairs
        # fed to einsum. Only one gradient will be computed for a
        # unique tensor-label pair
        self._cache: Optional[Counter] = None

    def forward(self, *tensors) -> NDArray:
        self.save_for_backward(*tensors)
        in_labels = ','.join(self.in_labels)
        return np.einsum(
            '->'.join((in_labels, self.out_labels)),
            *(t.data for t in tensors),
            optimize=self.optimize,
        ) 

    @property
    def cache(self) -> Counter:
        if self._cache is None:
            # This is hacky, but because this caching mechanism depends on the 
            # tensor-ids, we have to build the cache here - in case einsum is 
            # used to produce a view involved in an inplace operations, and 
            # placeholder tensors need be replaced.
            self._cache = Counter([
                (id(p),label) for p, label in zip(self.parents, self.in_labels)
            ])
        return self._cache

    def _backward(self, partial: NDArray, index: int):
        in_labels = self.in_labels.copy()
        original_parent_label = in_labels.pop(index)
        parent = self.parents[index]

        factor = self.cache[(id(parent), original_parent_label)]
        if factor == 0:
            # the gradient for the current tensor-label pair
            # has already been computed, scaled, and back-propped,
            # skip gradient calculation.
            return 

        parents_data = tuple(i.data for i in self.parents)
        self.cache[(id(parent), original_parent_label)] = 0

        parent_label = unique_from_end(original_parent_label)
        repeated_labels = len(parent_label) != len(original_parent_label)

        if repeated_labels:
            # example fwd-prop: einsum("iji -> ij", x)
            # "iji" becomes "ji", later we will write along
            # the diagonal of an array to reinstate this axis that
            # we just removed
            mapping_gen = tuple(
                {k: v for k, v in zip(label, arr.shape)}
                for label, arr in zip(self.in_labels, parents_data)
            )
            label_to_size = merge_max_mappings(*mapping_gen)
            parent_shape = tuple(label_to_size[label] for label in parent_label)

        else:
            parent_shape = parent.shape

        # ji
        partial_label = self.out_labels

        # Catch indices over which un-contracted sum was performed
        # for the given variable: e.g for var-0 in "ijk, jk -> k"
        # i is summed over without contraction with another tensor
        #
        # Backpropping through this is illegal, as it requires the creation
        # of an axis; e.g. k, jk -> ijk
        # Broadcast the gradient along all such dimensions; e.g. k -> ik
        # then proceed as usual; e.g. ik, jk -> ijk
        unique_in_labels = set(chain.from_iterable(in_labels))
        unique_in_labels |= set(partial_label)
        any_sumreduced_axis = set(parent_label) - unique_in_labels

        if any_sumreduced_axis:
            exp_dims = [slice(None)] * partial.ndim 
            partial_shape = list(partial.shape)
            partial_label = list(partial_label)

            for n, label in enumerate(parent_label):
                if label not in unique_in_labels:
                    partial_label.insert(n, label) 
                    exp_dims.insert(n, np.newaxis)
                    partial_shape.insert(n, parent_shape[n])

            partial = np.broadcast_to(
                partial if not partial.ndim else partial[tuple(exp_dims)], 
                partial_shape
            )
            partial_label = ''.join(partial_label)

        # "ji, k -> ijk"
        back_prop_labels = ",".join([partial_label] + in_labels) + "->" + parent_label
        # (partial, y)
        operands = (partial,) + parents_data[:index] + parents_data[index+1:]

        if not repeated_labels:
            # dfdx: einsum("ji, k -> ijk", partial, y)
            outshape = parent.shape
            dfdx = collapse(
                np.einsum(back_prop_labels, *operands, optimize=self.optimize), 
                outshape
            )
            if parent_shape != dfdx.shape:
                # if y was broadcast over x, the gradient needs to
                # be broadcast to x's shape: dfdx-shape (i,j,1) -> (i,j,k)
                dfdx = np.broadcast_to(dfdx, parent_shape)
            if factor > 1:
                # This tensor-label pair appears several times as
                # input to einsum. Scale the gradient accordingly
                # such that the full contribution of the tensor-label
                # pair is accounted for.
                dfdx *= factor
            parent.grad += dfdx
            return

        # Accommodate trace by writing to strided view on array of zeros
        # For example:
        #
        # fwd:  einsum('ijkji, k -> jk', x, y)
        # dfdx: einsum('jk, k -> kji', grad, y, out=view_of_x)
        #
        # writing to `view_of_x`, which is a view along the appropriate
        # diagonals of x, is equivalent to:
        #
        # dfdx: einsum('jk, k -> ijkji', grad, y)
        #
        # which is formally correct but not supported by einsum.
        dfdx = np.zeros(tuple(label_to_size[i] for i in original_parent_label))
        out_view_shape = tuple(label_to_size[i] for i in parent_label)

        # compute strides required to traverse the appropriate diagonals of
        # the output tensor.
        strides = tuple(
            sum(dfdx.strides[ind] for ind in get_indices(label, original_parent_label))
            for label in parent_label
        )
        out_view = as_strided(dfdx, shape=out_view_shape, strides=strides)
        np.einsum(back_prop_labels, *operands, out=out_view, optimize=self.optimize)
        
        if factor > 1:
            # This tensor-label pair appears several times as
            # input to einsum. Scale the gradient accordingly
            # such that the full contribution of the tensor-label
            # pair is accounted for.
            dfdx *= factor

        parent.grad += dfdx


    def backward(self, partial: NDArray):
        # for example:
        # fwd:           "ijk, k -> ji",    x, y
        # bkwd (var: 0): "ji, k -> ijk", grad, y
        # bkwd (var: 1): "ji, ijk -> k", grad, x
        for i, p in enumerate(self.parents):
            if p.requires_grad:
                self._backward(partial, i)


def einsum(
        subscripts: str, 
        *operands: Union[Tensor, NDArray],
        optimize: Optimize = False
    ) -> Tensor:
    in_labels, out_labels, _ = _parse_einsum_input([subscripts, *operands])
    return Tensor.comm(Einsum(in_labels, out_labels, optimize), *operands)
