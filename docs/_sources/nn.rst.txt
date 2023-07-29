.. currentmodule:: giagrad.nn

giagrad.nn
==========

Cotainers
~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: class.rst

    Module
    Sequential

Linear Layers
~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: class.rst

    Linear

Dropout Layers
~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: class.rst

    Dropout
    DropoutND


Convolution Layers
~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: class.rst

    Conv1D
    Conv2D
    Conv3D

Normalization Layers
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: class.rst

    BatchNormND
    LayerNorm

Loss Functions
~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: class.rst

    CrossEntropyLoss


Activations
~~~~~~~~~~~

Unlike the activation functions already implemented as methods in the 
:class:`giagrad.Tensor`, there are also classes that, on their own, have 
the same behavior as their homologous methods in :class:`giagrad.Tensor`.

Thet are useful when creating modules such as Sequential:

.. code-block::
    
    import giagrad.nn as nn
    model = nn.Sequential(
            nn.Linear(128, 40),
            nn.LeakyReLU(neg_slope=3),
            nn.Dropout(0.4),
            nn.Linear(40, 10)
        )

They behave like callable classes:

.. code-block::

    activation = nn.SiLU(alpha=0.5)
    t = Tensor.empty(2, 3).uniform(-10, 10)
    activation(t)


.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: class.rst

    ReLU
    ReLU6
    Hardswish
    Sigmoid
    ELU
    SiLU
    Tanh
    LeakyReLU
    SoftPlus
    Mish
    GELU
    QuickGELU
    Softmax
    LogSoftmax