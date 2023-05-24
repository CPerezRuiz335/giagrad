import giagrad as gg
from giagrad.shapeops import _Padding

t = gg.Tensor.empty(2, 3, 4, dtype=int, requires_grad=True).uniform(b=10)
print(t)
p = t.padding((5, (1, 0), 2,2))
print('sliced',p[slice(1, None), slice(2,-2), slice(2, -2)])
print(p)
print('padding', p.fn.padding)

p.backward()
print(t.grad)
print(p.grad)
