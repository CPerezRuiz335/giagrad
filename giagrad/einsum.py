# https://github.com/rsokl/MyGrad/blob/master/src/mygrad/linalg/ops.py
from itertools import chain
from collections import Counter
from functools import reduce
from typing import Literal, Tuple, Union, List

import numpy as np
from numpy.typing import NDArray
from numpy.lib.stride_tricks import as_strided

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
        self.in_labels = in_labels
        self.out_labels = out_labels
        self.optimize = optimize    

        # cache counts the number of redundant tensor-label pairs
        # fed to einsum. Only one gradient will be computed for a
        # unique tensor-label pair
        self._cache = None

    def forward(self, *tensors) -> NDArray:
        self.save_for_backward(*tensors)
        return np.einsum(
            '->'.join((self.in_labels, self.out_labels)),
            *(t.data for t in tensors),
            optimize=self.optimize,
        ) 

    @property
    def cache(self) -> Counter:
        if self._cache is None:
            # This is hacky, but because this caching mechanism depends on the tensor-ids,
            # we have to build the cache here - in case einsum is used to produce a view
            # involved in an inplace operations, and placeholder tensors need be replaced.
            #
            # Creating the cache in __call__ could create a nasty inconsistency between
            # tensor ids
            split_in_labels = self.in_labels.replace(' ', '').split(',')
            self._cache = Counter([
                (id(p),label) for p, label in zip(self.parents, split_in_labels)
            ])
        return self._cache

    def _backward(self, partial: NDArray, index: int):
        self.out_lbls = self.out_labels
        self.in_lbls = self.in_labels.replace(' ', '').split(',')
        in_lbls = self.in_lbls.copy()
        original_var_lbl = in_lbls.pop(index)
        var = self.parents[index]
        print(f"{self.cache = }")
        print('asfasdf', f"{(id(var), original_var_lbl) = }")

        factor = self.cache[(id(var), original_var_lbl)]
        if factor == 0:
            # the gradient for the current tensor-label pair
            # has already been computed, scaled, and back-propped,
            # skip gradient calculation.
            print(f"{factor = }")
            print(f"{self.cache = }")
            return 

        numpy_arrays = tuple(i.data for i in self.parents)
        self.cache[(id(var), original_var_lbl)] = 0

        var_lbl = unique_from_end(original_var_lbl)
        repeat_lbls = len(var_lbl) != len(original_var_lbl)

        if repeat_lbls:
            # example fwd-prop: einsum("iji -> ij", x)
            # "iji" becomes "ji", later we will write along
            # the diagonal of an array to reinstate this axis that
            # we just removed
            mapping_gen = tuple(
                {k: v for k, v in zip(lbl, arr.shape)}
                for lbl, arr in zip(self.in_lbls, numpy_arrays)
            )
            print(f"{mapping_gen = }")
            lbl_to_size = merge_max_mappings(*mapping_gen)
            print(f"{lbl_to_size = }")
            var_shape = tuple(lbl_to_size[lbl] for lbl in var_lbl)
            print(f"{var_shape = }")

        else:
            var_shape = self.parents[index].shape

        # ji
        partial_lbl = self.out_lbls

        # Catch indices over which un-contracted sum was performed
        # for the given variable: e.g for var-0 in "ijk, jk -> k"
        # i is summed over without contraction with another tensor
        #
        # Backpropping through this is illegal, as it requires the creation
        # of an axis; e.g. k, jk -> ijk
        # Broadcast the gradient along all such dimensions; e.g. k -> ik
        # then proceed as usual; e.g. ik, jk -> ijk
        unique_in_lbls = set(chain.from_iterable(in_lbls)) | set(partial_lbl)
        if len(set(var_lbl) - unique_in_lbls) > 0:
            exp_dims = [slice(None) for i in range(partial.ndim)]
            partial_shape = list(partial.shape)
            for n, lbl in enumerate(var_lbl):
                if lbl not in unique_in_lbls:
                    partial_lbl = partial_lbl[:n] + lbl + partial_lbl[n:]
                    exp_dims.insert(n, np.newaxis)
                    partial_shape.insert(n, var_shape[n])
            print(f"{partial_lbl = }")
            partial = np.broadcast_to(
                partial if not partial.ndim else partial[tuple(exp_dims)], partial_shape
            )

        # "ji, k -> ijk"
        back_prop_lbls = ",".join([partial_lbl] + in_lbls) + "->" + var_lbl

        # (partial, y)
        operands = (partial,) + numpy_arrays[:index] + numpy_arrays[index + 1 :]

        if not repeat_lbls:
            # dfdx: einsum("ji, k -> ijk", partial, y)
            outshape = self.parents[index].shape
            dfdx = collapse(
                np.einsum(back_prop_lbls, *operands, optimize=self.optimize), 
                outshape
            )
            if var_shape != dfdx.shape:
                # if y was broadcast over x, the gradient needs to
                # be broadcast to x's shape: dfdx-shape (i,j,1) -> (i,j,k)
                dfdx = np.broadcast_to(dfdx, var_shape)
            if factor > 1:
                # This tensor-label pair appears several times as
                # input to einsum. Scale the gradient accordingly
                # such that the full contribution of the tensor-label
                # pair is accounted for.
                dfdx *= factor
            var.grad += dfdx; return

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
        dfdx = np.zeros(tuple(lbl_to_size[i] for i in original_var_lbl))
        out_view_shape = tuple(lbl_to_size[i] for i in var_lbl)

        # compute strides required to traverse the appropriate diagonals of
        # the output tensor.
        strides = tuple(
            sum(dfdx.strides[ind] for ind in get_indices(lbl, original_var_lbl))
            for lbl in var_lbl
        )
        out_view = as_strided(dfdx, shape=out_view_shape, strides=strides)
        print(f"{in_lbls = }")
        print(f"{self.out_lbls = }")
        print(f"{back_prop_lbls = }")
        print(f"{[t.shape for t in operands] = }")
        print(f"{out_view.shape = }")

        np.einsum(back_prop_lbls, *operands, out=out_view, optimize=self.optimize)
        if factor > 1:
            # This tensor-label pair appears several times as
            # input to einsum. Scale the gradient accordingly
            # such that the full contribution of the tensor-label
            # pair is accounted for.
            dfdx *= factor
        var.grad += dfdx; return


    def backward(self, partial: NDArray):
        # for example:
        # fwd:           "ijk, k -> ji",    x, y
        # bkwd (var: 0): "ji, k -> ijk", grad, y
        # bkwd (var: 1): "ji, ijk -> k", grad, x
        for i, p in enumerate(self.parents):
            if p.requires_grad:
                print("llamo a _backward")
                print(f"{p.shape = }")
                self._backward(partial, i)


def einsum(
        subscripts: str, 
        *operands: Union[Tensor, NDArray],
        optimize: Optimize = False
    ) -> Tensor:
    in_labels, out_labels = subscripts.split('->')
    return Tensor.comm(Einsum(in_labels, out_labels, optimize), *operands)
