from __future__ import annotations
from giagrad.tensor import Tensor, Context
from giagrad.mlops import LogSoftmax
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, Tuple
import math

class CrossEntropyLoss(Context):

	def __init__(self, reduction: str = 'sum'):
		self.reduction = reduction

	def __save_for_backward(self, w: Tensor, one_hot: NDArray, log_softmax: NDArray):
		super().__init__((w,)) 
		self.one_hot = one_hot
		self.softmax = np.exp(log_softmax)
		return self

	@classmethod
	def forward(cls, weights, y: Union[Tensor, NDArray], axis: int, reduction: str = 'sum') -> Tuple[NDArray, CrossEntropyLoss]:
		y = y.data.flatten() if isinstance(y, Tensor) else y.flatten()
		log_softmax, _ = LogSoftmax.forward(weights, axis=axis)	
		# one hot
		one_hot = np.zeros_like(weights.data)
		one_hot[np.arange(y.size), y] = 1

		return -one_hot * log_softmax, cls(reduction).__save_for_backward(weights, one_hot, log_softmax)

	def __call__(self, weights: Tensor, y: Union[Tensor, NDArray], axis: int = 1) -> Tensor:
		# weights are unnormalized logits
		# y must be a sequence of classes numerically encoded from 0 to C-1
		if self.reduction == 'sum':
			return Tensor.comm(CrossEntropyLoss, weights, y=y, axis=axis, reduction=self.reduction).sum()
		if self.reduction == 'mean':
			return Tensor.comm(CrossEntropyLoss, weights, y=y, axis=axis, reduction=self.reduction).mean()

	def backward(self, partial: NDArray):
		p = self.parents[0]
		if p.requires_grad:
			p.grad += partial * (self.softmax - self.one_hot)

	def __str__(self):
		return f"CrossEntropyLoss(reduction = {self.reduction})"