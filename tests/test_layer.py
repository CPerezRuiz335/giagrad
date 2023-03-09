import sys; sys.path.append('../')
from giagrad.nn.layers import Linear

import numpy as np
l = Linear(10, 3)
print('parmeters', l.parameters())
print()
print('__odict__', l.__odict__)
print()
print('w, b')
print(l.w, '\n', l.b)
print()
X = np.ones((1, 10), dtype=np.float32).T
print(X)
print('asfasdf',l.w.grad)
out = l(X).relu()
print(out.backward())
print('asfasdf',l.w.grad)
print(out)