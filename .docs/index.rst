.. giagrad documentation master file, created by
   sphinx-quickstart on Tue Mar 28 22:15:53 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _giagrad: https://github.com/CPerezRuiz335/giagrad
.. _NumPy: https://numpy.org/
.. _standard library: https://docs.python.org/3/library/
.. _PyTorch: https://pytorch.org/
.. _tinygrad: https://github.com/geohot/tinygrad
.. _numpy-ml: https://github.com/ddbourgin/numpy-ml
.. _demo: https://github.com/CPerezRuiz335/giagrad/blob/main/Demo_MNIST.ipynb

GIAGRAD DOCUMENTATION
=====================

`giagrad`_ is a deep learning framework written in `NumPy`_ made by and for students.

This library aims to serve as a reference implementation for those interested 
in gaining a strong understanding of deep learning principles and as a 
place to develope skills such as GitHub, software development, documentation, 
etc. With that in mind, giagrad is designed to offer fast prototyping and 
experimentation with the spirit of `numpy-ml`_ and a syntax very similar, 
if not identical, to `PyTorch`_ or `tinygrad`_. 


This example should give you an idea:
     
.. code-block::

   import giagrad.nn as nn
   from giagrad.optim import SGD
   
   class MLP(nn.Module):
       def __init__(self):
           super().__init__()
           
           self.l1 = nn.Linear(784, 550)
           self.l2 = nn.Linear(550, 10)
           self.dropout = nn.Dropout(0.2)
           
       def __call__(self, x):
           x = self.l1(x).relu()
           x = self.dropout(x)
           x = self.l2(x)
           return x
   
   model = MLP()
   optimizer = SGD(model.parameters(), lr=0.001)
   criterion = nn.CrossEntropyLoss(reduction='mean')
   model.train()
   
    # ... follow as you would with pytorch, with (X, y) data
    
   optimizer.zero_grad() 
   output = model(X)
   loss = criterion(output, y)
   loss.backward()
   optimizer.step()

   model.eval()
  
(You can find this example extended in `demo`_ for MNIST dataset)

giagrad doesn't aim to be blazingly fast or professional, but rather well-made, 
concise, and well-documented. Therefore, read the docs but also read the code, 
since its philosophy could be based on:

   If it can't be understood, maybe it's not worth programming.

In this way, the documentation strives for transparency, as there is no fear of changing, 
refactoring, or redoing the code if it can be improved. 

This documentation includes a more theoretical section `Learn`_ and the `Python API`_.

.. warning:: 
   Both the software and documentation are works in progress, and are likely 
   to have typos, bugs, and poorly worded sections. If you come across any 
   of these issues, please let us know by filing an issue or submitting a 
   pull request.

   Please note that this software is provided "as-is", with no guarantees that 
   it will meet your needs or be free of bugs. Use it at your own risk!

.. toctree::
   :maxdepth: 1
   :caption: Learn
   :name: Learn

   autograd

.. toctree::
   :maxdepth: 1
   :caption: Python API
   :name: Python API

   giagrad  
   nn
   nn_activations
   optim
   display

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`