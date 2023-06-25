import sys
sys.path.append('../')

from typing import *
from numpy.typing import NDArray
from time import time 
from giagrad import Tensor
import numpy as np
import giagrad.nn as gnn

from timeit import timeit

def test(x):
    conv1 = gnn.Conv2D(
        out_channels=32,
        kernel_size=(2, 2),
        bias=False
    )

    conv2 = gnn.Conv2D(
        out_channels=32,
        kernel_size=(2, 2),
        bias=False
    )

    conv3 = gnn.Conv2D(
        out_channels=32,
        kernel_size=(2, 2),
        bias=False
    )

    conv4 = gnn.Conv2D(
        out_channels=32,
        kernel_size=(2, 2),
        bias=False
    )

    for _ in range(5):
        y = conv1(x)
        y = conv2(y)
        y = conv3(y)
        y = conv4(y)
        y.backward()



if __name__ == "__main__":
    # tensordot 99.01102873799937
    x = Tensor.empty(1000, 3, 28, 28)
    print(f"{timeit(lambda: test(x), number=10) = }") 
