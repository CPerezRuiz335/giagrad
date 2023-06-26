.. currentmodule:: giagrad
.. _numpy.array: https://numpy.org/doc/stable/reference/generated/numpy.array.html
.. _micrograd: https://github.com/karpathy/micrograd

Autograd
========

An autograd engine is a tool that automatically compute gradients (i.e., the 
vector of partial derivatives of the function with respect to its inputs) of a 
target function. This is important because gradient computations are at the core 
of optimization algorithms, which are used to adjust the parameters of a machine 
learning model to minimize the difference between the predicted and actual values. 

An autograd engine can work with scalar-valued functions like `micrograd`_ and 
with more complex data structures. Conceptually it can be visualized like a 
computational graph, like the one below: 


.. image:: images/micrograd_light.svg
	:target: https://github.com/karpathy/micrograd
	:height: 200
	:width: 1400

|

In the context of machine learning, the target function is the loss function, 
which is normally the final function applied to the computational graph, and the 
parameters are also called weights or data. However, these parameters must be 
contained in some data structure that is convenient and optimal, which is why 
tensors are used.

Therefore, a decent autograd tensor library is almost all that is needed to build 
neural networks and start doing deep learning. In giagrad the autograd engine is 
built on top of :class:`giagrad.Tensor` and :class:`giagrad.tensor.Function` 
classes, which are the base of giagrad. In reality, giagrad could work only with 
:class:`giagrad.Tensor`, but to mantain modularity and keeping the code more 
readable, :class:`giagrad.tensor.Function` exists.

:class:`giagrad.Tensor` is the data structure, which is based on a `numpy.array`_ 
for simplicity, and the computational graph is done with 
:class:`giagrad.tensor.Function`. To add new functionalities to 
:class:`giagrad.Tensor`, such as a new activation function, one only needs to
create that activation function class and add a new method to 
:class:`giagrad.Tensor`, see :meth:`giagrad.Tensor.comm`.

``giagrad.mlops.ReLU`` source code should give an idea:

.. code-block::

	class ReLU(Function):
	    def __init__(self):
	        super().__init__()

	    def forward(self, t1) -> NDArray:
	        self.save_for_backward(t1)
	        return np.maximum(t1.data, 0) 

	    def backward(self, partial: NDArray):
	        p = self.parents[0]
	        if p.requires_grad:
	            p.grad += partial * (p.data > 0).astype(int)