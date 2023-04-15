.. currentmodule:: giagrad
.. _numpy.array: https://numpy.org/doc/stable/reference/generated/numpy.array.html

giagrad
=======

CORE
++++

:class:`giagrad.Tensor` and :class:`giagrad.tensor.Context` constitute the base of giagrad.

:class:`giagrad.Tensor` can be initialized with an *array_like* object, in fact you can create a tensor
out of everything `numpy.array`_ constructor accepts

>>> Tensor(range(10))                                                         
tensor: [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
>>> Tensor([[1, 2, 1], [3, 4, 3]])
tensor: [[1. 2. 1.]
         [3. 4. 3.]]

By default every tensor's data is ``float32`` but it can be modified

>>> Tensor(range(10), dtype=np.int8)
tensor: [0 1 2 3 4 5 6 7 8 9]

For some specific initialization such as :func:`~Tensor.xavier_normal`, you should create an
empty tensor and apply the in-place initialization that you want, see :func:`~Tensor.empty` and `Initializers`_

.. code-block::

    >>> Tensor.empty(2, 2, 4).xavier_normal()
    tensor: [[[-0.21414495  0.38195378 -1.3415855  -1.0419445 ]
              [ 0.2715997   0.428172    0.42736086  0.14651838]]

             [[ 0.87417895 -0.56151503  0.4281528  -0.65314466]
              [ 0.69647044  0.25468382 -0.08594387 -0.8892542 ]]]


Context
-------

.. class:: giagrad.tensor.Context

    Abstract class for all Tensor operators.
    
    Operators extend the Tensor class to provide additional 
    functionality. The Context class behavior is accessed through the 
    :func:`~giagrad.Tensor.comm` [1]_ method. To mantain modularity,
    the operators are implemented in separate files.

    For developer use.

    :ivar parents: 
        Tensor/s needed for the child class that inherits Context. 
        :attr:`~parents` **must not** contain other types than Tensor, if 
        other attributes are needed they should be an instance variable, 
        e.g. :math:`\text{neg_slope}` variable for Leaky ReLU.
    :vartype parents: Tensor

    :ivar _name: 
        Complex modules that want to override the representation of the 
        output tensor may find it useful to modify the _name attribute.
    :vartype _name: str, optional


.. autosummary::
    :toctree: generated
    :nosignatures:

    giagrad.tensor.Context.forward
    giagrad.tensor.Context.backward


Tensor class reference
----------------------

.. class:: giagrad.Tensor
    
    Autodifferentiable multi-dimensional array and the core of giagrad.
    
    Tensor extends the functionality of a `numpy.array`_ implicitly creating 
    an autoddiferentiable computational graph with the help of :class:`giagrad.tensor.Context`.
    An instance is only differentiable iff it has a Context and requires_grad [1]_. 
    The name is optional, just for :ref:`giagrad.display`.

    :ivar data: Weights of the tensor.
    :vartype data: array_like

    :ivar requires_grad: If ``True`` makes tensor autodifferentiable.
    :vartype requires_grad: bool, default: False

    :ivar name: Optional name of the tensor. For display purpose.
    :vartype name: str, optional

    :ivar dtype: Data type of the ``.data``
    :vartype dtype: np.float32

.. [1]
    See :ref:`Autograd`.

Attributes
~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:

    Tensor.T
    Tensor.shape
    Tensor.dtype
    Tensor.size
    Tensor.ndim

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