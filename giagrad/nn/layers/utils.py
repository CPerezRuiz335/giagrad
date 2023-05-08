from typing import Tuple, Union
from itertools import chain, groupby
from collections.abc import Sized

def flat_tuple(tup: Tuple[Union[Tuple[int, ...], int], ...]) -> Tuple[int, ...]:
    """flat a tuple made of int or tuples of int"""
    return tuple(chain(*(i if isinstance(i, tuple) else (i,) for i in tup)))

def same_len(*args: Sized):
    """check if all input parameters have same length"""
    g = groupby(args, lambda x: len(x))
    return next(g, True) and not next(g, False)

def format_tuples(
        padding: Tuple[Union[Tuple[int, ...], int], ...]
    ) -> Tuple[Tuple[int, int], ...]:
    return tuple(i if isinstance(i, tuple) else (i, i) for i in padding)

