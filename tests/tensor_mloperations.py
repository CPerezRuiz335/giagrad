import torch # type: ignore
import sys; sys.path.append('../')
from  giagrad.tensor import Tensor

def test_ops():
	a = Tensor([1.5, -3.4, 3.0, 1.2, 0, 100.333], requires_grad=True)
	b = a.relu()
	c = b.sigmoid()
	d = c.elu()
	e = d.silu()
	f = e.tanh()
	g = f.leakyrelu()
	h = g.softplus()
	i = h.relu6()
	j = i.mish()
	k = j.hardswish()
	z = k.sum()
	z.backward()
	amg, zmg = a, z

	a = torch.Tensor([1.5, -3.4, 3.0, 1.2, 0 , 100.333])
	a.requires_grad = True
	b = a.relu()
	c = b.sigmoid()
	d = torch.nn.functional.elu(c)
	e = torch.nn.functional.silu(d)
	f = e.tanh()
	g = torch.nn.functional.leaky_relu(f)
	h = torch.nn.functional.softplus(g)
	i = torch.nn.functional.relu6(h)
	j = torch.nn.functional.mish(i)
	k = torch.nn.functional.hardswish(j)
	z = k.sum()
	z.backward()
	apt, zpt = a, z

	tol = 1e-6

	# forward pass went well
	assert abs(zmg.data - zpt.data.item()) < tol
	# backward pass went well
	assert all(abs(amg.grad - apt.grad.detach().numpy()) < tol) 


def test_more_ops():
	a = Tensor([1.5, -3.4, 3.0, 1.2, 0, -100.0], requires_grad=True)
	b = a.gelu()
	z = b.sum()
	z.backward()
	amg, zmg = a, z

	a = torch.Tensor([1.5, -3.4, 3.0, 1.2, 0 ,-100.0])
	a.requires_grad = True
	b = torch.nn.functional.gelu(a)
	z = b.sum()
	z.backward()
	apt, zpt = a, z

	tol = 1e-3

	# forward pass went well
	assert abs(zmg.data - zpt.data.item()) < tol
	# backward pass went well
	assert all(abs(amg.grad - apt.grad.detach().numpy()) < tol) 


def test_more_more_ops():
	a = Tensor([[1.5, -3.4, 3.0, 1.2, 0 ,-100.0]], requires_grad=True).T
	b = a.softmax()
	c = b.log_softmax()
	z = c.sum()
	z.backward()
	amg, zmg = a, z

	a = torch.Tensor([[1.5, -3.4, 3.0, 1.2, 0 ,-100.0]]).t()
	a.requires_grad = True
	b = torch.nn.functional.softmax(a, dim=0)
	b.retain_grad()
	c = torch.nn.functional.log_softmax(b, dim=0)
	
	z = c.sum()
	z.backward()
	apt, zpt = a, z

	tol = 1e-6

	# forward pass went well
	assert abs(zmg.data - zpt.data.item()) < tol
	# backward pass went well
	assert all(abs(amg.grad - apt.grad.detach().numpy()) < tol) 

if __name__ == "__main__":
	test_ops()
	test_more_ops()
	test_more_more_ops()