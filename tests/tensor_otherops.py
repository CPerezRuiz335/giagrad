import torch # type: ignore
import numpy as np
import sys; sys.path.append('../')
from giagrad.tensor import Tensor

def test_einsum():
	pa = torch.rand(3, 5, 2, 6, requires_grad=True) 
	pb = torch.rand(3, 6, 2, 6, 3, requires_grad=True)
	pc = torch.rand(6, 2, 7, 3, requires_grad=True)

	apt = torch.einsum("abcd, adcda -> abc", pa, pb)
	bpt = torch.einsum("adcda, dcha -> h", pb, pc)
	cpt = torch.einsum("abcd, adcda, dcha -> abh", pa, pb, pc)

	apt.sum().backward()
	bpt.sum().backward()
	cpt.sum().backward() 

	na, nb, nc = pa.detach().numpy(), pb.detach().numpy(), pc.detach().numpy()
	ga = Tensor(na, requires_grad=True) 
	gb = Tensor(nb, requires_grad=True)
	gc = Tensor(nc, requires_grad=True)

	amg = ga.einsum("abcd, adcda -> abc", gb)
	bmg = gb.einsum("adcda, dcha -> h", gc)
	cmg = ga.einsum("abcd, adcda, dcha -> abh", gb, gc)

	amg.sum().backward()
	bmg.sum().backward()
	cmg.sum().backward() 

	tol = 1e-5
	assert np.all(abs(ga.grad - pa.grad.detach().numpy()) < tol)
	assert np.all(abs(gb.grad - pb.grad.detach().numpy()) < tol)
	assert np.all(abs(gc.grad - pc.grad.detach().numpy()) < tol)

if __name__ == "__main__":
	test_einsum()
