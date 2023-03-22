from __future__ import annotations
from giagrad.tensor import Tensor, Context
from giagrad.mlops import LogSoftmax
from giagrad.reductionops import Sum
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, Tuple
import math
from functools import wraps

class CrossEntropy(Context):
	def __init__(self, *tensor, one_hot: NDArray, log_softmax: NDArray):
		self.one_hot = one_hot
		self.softmax = np.exp(log_softmax)
		super().__init__(tensor)

	@classmethod
	def forward(cls, t1, y: NDArray, axis: int) -> Tuple[NDArray, CrossEntropy]:
		log_softmax, _ = LogSoftmax.forward(t1, axis=axis)	
		# one hot
		one_hot = np.zeros_like(t1.data)
		one_hot[np.arange(y.size), y] = 1

		return -one_hot * log_softmax, cls(t1, one_hot=one_hot, log_softmax=log_softmax)

	def backward(self, partial: NDArray):
		p = self.parents[0]
		if p.requires_grad:
			p.grad += partial * (self.softmax - self.one_hot)

	def __str__(self):
		return f"CrossEntropy"

class CrossEntropyLoss:

	def __init__(self, reduction: str = 'mean'):
		self.reduction = reduction

	def __call__(self, weights: Tensor, y_true: Union[Tensor, NDArray], axis: int = 1) -> Tensor:
		# weights are unnormalized logits
		# y must be a sequence of classes numerically encoded from 0 to C-1
		y_true = y_true.data if isinstance(y_true, Tensor) else y_true
		if self.reduction == 'sum':
			t = Tensor.comm(CrossEntropy, weights, y=y_true, axis=axis).sum()
		if self.reduction == 'mean':
			t = Tensor.comm(CrossEntropy, weights, y=y_true, axis=axis).mean(dim=0).sum()
		t._ctx._name = f"CrossEntropyLoss(reduction = {self.reduction})"
		return t	