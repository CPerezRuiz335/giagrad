import sys
sys.path.append('../')
from giagrad import Tensor
import numpy as np
import giagrad.nn as gnn
import torch
import torch.nn as tnn

def test_batchnorm():
	at = torch.rand(3, 4, 5, 6, requires_grad=True)
	ag = Tensor(at.detach().numpy().copy(), requires_grad=True)

	torch_layer = tnn.BatchNorm2d(num_features=4)
	torch_out = torch_layer(at)
	torch_out.sum().backward()

	gg_layer = gnn.BatchNormND()
	gg_out = gg_layer(ag)
	gg_out.sum().backward()

	assert np.allclose(gg_out.data, torch_out.detach().numpy(), atol=1e-6)
	assert np.allclose(ag.grad, at.grad.detach().numpy(), atol=1e-5)

	torch_layer.eval()
	torch_out = torch_layer(at)
	torch_out.sum().backward()

	gg_layer.eval()
	gg_out = gg_layer(ag)
	gg_out.sum().backward()

	assert np.allclose(gg_out.data, torch_out.detach().numpy(), atol=1e-4)
	assert np.allclose(ag.grad, at.grad.detach().numpy(), atol=1e-4)


def test_layer_norm():
	at = torch.rand(20, 5, 10, 10, requires_grad=True)
	ag = Tensor(at.detach().numpy().copy(), requires_grad=True)

	torch_layer = tnn.LayerNorm([5, 10, 10])
	torch_out = torch_layer(at)
	torch_out.sum().backward()

	gg_layer = gnn.LayerNorm(dimensions=3)
	gg_out = gg_layer(ag)
	gg_out.sum().backward()

	assert np.allclose(gg_out.data, torch_out.detach().numpy(), atol=1e-6)
	assert np.allclose(ag.grad, at.grad.detach().numpy(), atol=1e-5)

if __name__ == "__main__":
	test_batchnorm()
	test_layer_norm()