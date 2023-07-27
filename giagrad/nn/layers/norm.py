from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from giagrad.tensor import Tensor, Function
from giagrad.nn.containers import Module

class BatchNormND(Module):
	r"""
	Applies Batch Normalization as described in `Batch Normalization: 
	Accelerating Deep Network Training by Reducing Internal Covariate 
	Shift <https://arxiv.org/abs/1502.03167>`__ .
	

	Parameters
	----------
    eps: default: 1e-5
    	A value added to the denominator for numerical stability.
    momentum: default: 0.1
    	The value used for the running_mean and running_var computation. 
    	Can be set to ``None`` for cumulative moving average (i.e. simple 
    	average). 
    affine: default ``True``
    	A boolean value that when set to ``True``, this module has
        learnable affine parameters (:math:`\gamma` and :math:`\beta`). 
    track_running_stats: default: ``True`` 
    	A boolean value that when set to ``True``, this module tracks 
    	the running mean and variance, and when set to ``False``,
        this module does not track such statistics, in that case this 
        module always uses batch statistics in both training and eval 
        modes. 
	"""

	def __init__(
		self,
		eps: float = 1e-5,
		momentum: float = 0.1,
		affine: bool = True,
		track_running_stats: bool = True
	):	
		super().__init__()

		self.eps = eps
		self.momentum = momentum
		self.affine = affine
		self.track_running_stats = track_running_stats

		self.gamma: Tensor
		self.beta: Tensor
		self.running_mean: Tensor
		self.running_var: Tensor

	def __init_tensors(self, in_features: int):
		self.gamma = Tensor.empty(in_features, requires_grad=self.affine).ones()
		self.beta = Tensor.empty(in_features, requires_grad=self.affine).zeros()

		self.running_mean = Tensor.empty(in_features).zeros()
		self.running_var = Tensor.empty(in_features).zeros()

	def __call__(self, x: Tensor) -> Tensor:
		axis = x.shape[:0] + x.shape[2:]

		if self.training:
			mean = x.mean(axis, keepdims=True)
			var = x.var(axis, ddof=0, keepdims=True)

			if self.track_running_stats:
				self.running_mean *= self.momentum 
				self.running_mean += (1-self.momentum) * mean
				self.running_var *= self.momentum 
				self.running_var += (1-self.momentum) * var

			return (x-mean) / (var+self.eps).sqrt() * self.gamma + self.beta

		if not self.training:
			if self.track_running_stats:
				return (
					(x-self.running_mean) / (self.running_var+self.eps).sqrt() 
					* self.gamma 
					+ self.beta
				)
			else:
				mean = x.mean(axis, keepdims=True)
				var = x.var(axis, ddof=0, keepdims=True)

				return (x-mean) / (var+self.eps).sqrt() * self.gamma + self.beta


	def __str__(self):
		return (
            f"{type(self).__name__}("
            + f"eps={self.eps}, "
            + f"momentum={self.momentum}, "
            + f"affine={self.affine}"
            + (f", track_running_stats={self.track_running_stats}" 
            	if not self.track_running_stats else '')
            + ')'
        )
