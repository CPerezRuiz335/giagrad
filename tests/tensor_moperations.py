"""https://github.com/karpathy/micrograd/blob/master/test/test_engine.py"""
import torch # type: ignore
import sys; sys.path.append('../')
from  giagrad.tensor import Tensor
import numpy as np
from giagrad.display import draw_dot

def test_sanity_check():

    x = Tensor(-4.0, requires_grad=True)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xgg, ygg = x, y

    x = torch.Tensor([-4.0])
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ygg.data == ypt.data.item()
    # backward pass went well
    assert xgg.grad == xpt.grad.item()

def test_more_ops():

    a = Tensor(-4.0, requires_grad=True)
    b = Tensor(2.0, requires_grad=True).log()
    c = (a + b).reciprocal().abs()
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d.exp()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g = g.log()
    g.backward()
    agg, bgg, ggg = a, b, g

    a = torch.Tensor([-4.0])
    b = torch.Tensor([2.0]).log()
    a.requires_grad = True
    b.requires_grad = True
    c = (a + b).reciprocal().abs()
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d.exp()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g = g.log()
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(ggg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(agg.grad - apt.grad.item()) < tol
    assert abs(bgg.grad - bpt.grad.item()) < tol

def test_reductions():
    t = [[1.0, 2.0], [4.0, 4.0], [5.0, -16.0]]
    a = Tensor(t, requires_grad=True)
    b = a.sum()
    b.backward()
    agg, bgg = a, b

    a = torch.Tensor(t)
    a.requires_grad = True
    b = a.sum()
    b.backward()
    apt, bpt = a, b

    assert bgg.data == bpt.data.item()
    assert np.all(agg.grad == apt.grad.detach().numpy()) 

    a = Tensor(t, requires_grad=True)
    b = a.min()
    b.backward()
    agg, bgg = a, b

    a = torch.Tensor(t)
    a.requires_grad = True
    b = a.min()
    b.backward()
    apt, bpt = a, b

    assert bgg.data == bpt.data.item()
    assert np.all(agg.grad == apt.grad.detach().numpy()) 

    a = Tensor(t, requires_grad=True)
    b = a.max()
    b.backward()
    agg, bgg = a, b

    a = torch.Tensor(t)
    a.requires_grad = True
    b = a.max()
    b.backward()
    apt, bpt = a, b

    assert bgg.data == bpt.data.item()
    assert np.all(agg.grad == apt.grad.detach().numpy()) 

    a = Tensor(t, requires_grad=True)
    b = a.mean()
    b.backward()
    agg, bgg = a, b

    a = torch.Tensor(t)
    a.requires_grad = True
    b = a.mean()
    b.backward()
    apt, bpt = a, b

    assert bgg.data == bpt.data.item()
    assert np.all(agg.grad == apt.grad.detach().numpy()) 

def test_more_reductions():
    # test axis
    a = torch.randn(2, 3, 4, requires_grad=True)
    b = a.mean(dim=(0,1), keepdim=True) * a.sum(dim=(1, 2), keepdim=True)
    c = b.min()
    c.backward()
    apt, cpt = a, c
    
    a = Tensor(a.detach().numpy(), requires_grad=True)
    b = a.mean(axis=(0,1), keepdims=True) * a.sum(axis=(1, 2), keepdims=True)
    c = b.min()
    c.backward()
    agg, cgg = a, c

    tol = 1e-6

    assert (cgg.data - cpt.data.item()) < tol
    assert agg.grad.shape == apt.grad.detach().numpy().shape
    assert np.all(abs(agg.grad.flatten() - apt.grad.detach().numpy().flatten()) < tol) 

    # test axis
    a = torch.randn(2, 3, 4, requires_grad=True)
    b = torch.randn(4, 3, 1, requires_grad=True)
    aprime = a.mean(dim=(0,2), keepdim=False)
    bprime = b.max(dim=0, keepdim=True)[0]
    c = aprime.max() * bprime.min()
    c.backward()
    apt, bpt, cpt = a, b, c 
    
    a = Tensor(a.detach().numpy(), requires_grad=True)
    b = Tensor(b.detach().numpy(), requires_grad=True)
    aprime = a.mean(axis=(0,2), keepdims=False)
    bprime = b.max(axis=0, keepdims=True)
    c = aprime.max() * bprime.min()
    c.backward()
    agg, bgg, cgg = a, b, c

    tol = 1e-6

    # forward passed
    assert (cgg.data - cpt.data.item()) < tol
    # backward passed
    assert agg.grad.shape == apt.grad.detach().numpy().shape
    assert bgg.grad.shape == bpt.grad.detach().numpy().shape

    assert np.all(abs(agg.grad.flatten() - apt.grad.detach().numpy().flatten()) < tol) 
    assert np.all(abs(bgg.grad.flatten() - bpt.grad.detach().numpy().flatten()) < tol) 

if __name__ == "__main__":
    test_sanity_check()
    test_more_ops()
    test_reductions()
    test_more_reductions()

