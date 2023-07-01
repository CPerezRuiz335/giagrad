import sys
sys.path.append('../')

import numpy as np

from giagrad import Tensor, einsum

a = Tensor.empty(6, 4, 2, 6, requires_grad=True).uniform()
b = Tensor.empty(2, 6, 5, requires_grad=True).uniform()
c = einsum('dbcd, cda -> b', a, b)
c.backward()

print(a.grad)
print(b.grad)

print(f"{c = }")
print(f"{np.all(a.grad == 0) = }")
print(f"{np.all(b.grad == 0) = }")
