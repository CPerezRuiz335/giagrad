from giagrad.tensor import Tensor
from giagrad.mlops import LogSoftmax
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union
import math

class CrossEntropyLoss:

	def __init__(self,  reduction: Optional[str] = 'sum'):
		self.reduction = reduction

	def __call__(self, weights: Tensor, y: Union[Tensor, NDArray]) -> float:
		# weights are unnormalized logits
		# y must be a sequence of classes numerically encoded from 0 to C-1
		y = y.data if isinstance(y, Tensor) else y

		log_softmax, _ = LogSoftmax.forward(weights.data)
		# one hot
		one_hot = np.zeros_like(weights.data)
		one_hot[y, range(len(y))] = 1

		# save for backwards
		self.t = weights 
		self.one_hot = one_hot
		self.softmax = np.exp(log_softmax)

		match self.reduction:
			case 'sum':
				return (-one_hot * log_softmax).sum() 
			case 'mean':
				return (-one_hot * log_softmax).mean() 

	def backward(self):
		match self.reduction:
			case 'sum':
				grad = self.one_hot - self.softmax
			case 'mean':
				grad = math.prod(self.t.shape) * (self.one_hot - self.softmax)

		self.t.backward(grad_output=grad)




