import torch # type: ignore
import sys; sys.path.append('../')
import giagrad
import numpy as np
from random import uniform, randint
import giagrad

FEATURES=100
OBS=200

def test_loss():
    X_list = [[uniform(-10, 10) for _ in range(FEATURES)] for _ in range(OBS)]
    y_list = [randint(0, FEATURES-1) for _ in range(OBS)]
    
    # CrossentropyLoss
    X = giagrad.Tensor(X_list).requires_grad_()
    y = giagrad.Tensor(y_list, dtype=np.int8)
    criterion = giagrad.nn.CrossEntropyLoss(reduction='mean')
    loss = criterion(X, y)
    loss.backward()
    lgg, xgg = loss, X

    X = torch.Tensor(X_list).requires_grad_()
    y = torch.tensor(y_list)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = criterion(X, y)
    loss.backward()
    lpt, xpt = loss, X 

    tol = 1e-4
    # backward pass went well
    assert np.all(abs(xgg.grad.flatten() - xpt.grad.detach().numpy().flatten()) < tol)
    # forward pass went well
    assert abs(lgg.data - lpt.data.item()) < tol


if __name__ == "__main__":
    test_loss()