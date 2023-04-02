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

.. class:: Tensor(data, requires_grad=False, context: Context | None = None, 
    name='', dtype=np.float32)

    Autodifferentiable multi-dimensional array and the core of giagrad.
    
    Tensor extends the functionality of a `numpy.array`_ implicitly creating 
    an autoddiferentiable computational graph with the help of :class:`giagrad.tensor.Context`.
    An instance is only differentiable iff it has a Context and requires_grad [1]_. 
    The name is optional, just for display (TODO:link) module.

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
    Tensor.constant
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

Tensor also supports basic arithmetic operations, reverse ones and
in-place too. Here's an example that showcases several operations that are 
actually supported:

>>> from giagrad import Tensor
>>> a = Tensor([-4.0, 9.0])
>>> b = Tensor([[2.0], [-3.0]])
>>> c = (a + b) / (a * b) + b**3
>>> d = c * (2 + b + 1) / a
>>> c
tensor: [[  8.25       8.611111]
         [-27.583334 -27.222221]] grad_fn: Sum
>>> d
tensor: [[-10.3125      4.7839503]
         [  0.         -0.       ]] grad_fn: Div
>>> c @ d
tensor: [[ -85.078125   39.46759 ]
         [ 284.45312  -131.9573  ]] grad_fn: Matmul

.. note::
    in-place operations (``+=``, ``-=``, ...) only modify data in-place, they do not create a new
    instance of Tensor.

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