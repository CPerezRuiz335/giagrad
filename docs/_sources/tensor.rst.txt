.. currentmodule:: giagrad
.. _numpy.array: https://numpy.org/doc/stable/reference/generated/numpy.array.html

Core
====

Context
----------------------
.. autoclass:: giagrad.tensor.Context
    :members:
    :special-members: __str__


Tensor class reference
----------------------

.. class:: Tensor(data: ndarray, requires_grad: bool = False, context: Optional[Context] = None, 
    name: str = '', dtype = np.float32)
    
    Autodifferentiable multi-dimensional array and the core of giagrad.
    
    Tensor extends the functionality of a `numpy.array`_ implicitly creating 
    an autoddiferentiable computational graph with the help of :class:`giagrad.tensor.Context`.

    An instance is only differentiable iff it has a Context and requires_grad [1]_. 
    The name is optional, just for display (TODO:link) module and by default every
    Tensor has data type float32.

.. [1]
    See section documentation (TODO: link).

Attributes
~~~~~~~~~~

.. autoattribute:: Tensor.T
.. autoattribute:: Tensor.shape
.. autoattribute:: Tensor.dtype
.. autoattribute:: Tensor.size
.. autoattribute:: Tensor.ndim

Gradient
~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.backward
    Tensor.no_grad
    Tensor.requires_grad_

Class Methods
~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.comm
    Tensor.empty

Initializers 
~~~~~~~~~~~~


.. autofunction:: calculate_gain


.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.zeros
    Tensor.ones
    Tensor.full
    Tensor.normal
    Tensor.uniform
    Tensor.dirac
    Tensor.xavier_uniform
    Tensor.xavier_normal
    Tensor.kaiming_uniform
    Tensor.kaiming_normal
    Tensor.sparse
    Tensor.orthogonal


Math Ops
~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.sqrt
    Tensor.square
    Tensor.exp
    Tensor.log
    Tensor.reciprocal
    Tensor.abs
    Tensor.add
    Tensor.sub
    Tensor.mul
    Tensor.pow
    Tensor.matmul
    Tensor.div

Activation Functions 
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.relu
    Tensor.sigmoid
    Tensor.elu
    Tensor.silu
    Tensor.tanh
    Tensor.leakyrelu
    Tensor.softplus
    Tensor.quick_gelu
    Tensor.gelu
    Tensor.relu6
    Tensor.mish
    Tensor.hardswish
    Tensor.softmax
    Tensor.log_softmax

Reduction Ops
~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.mean
    Tensor.sum
    Tensor.max
    Tensor.min

Indexing, Slicing, Reshaping Ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.permute
    Tensor.transpose