import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import torch # type: ignore
import giagrad
import numpy as np

np.random.seed(0)

def test_inits(echo):
    # just test no errors
    # PyTroch and Numpy have different RNGs
    zeros = giagrad.Tensor.empty(3, 3).zeros()
    ones = giagrad.Tensor.empty(4, 7).ones()
    constant = giagrad.Tensor.empty(5, 8, requires_grad=True).constant(3.892)
    normal = giagrad.Tensor.empty(1, 3, requires_grad=True).normal(0, 3.4)
    uniform = giagrad.Tensor.empty(4, 4, requires_grad=True).uniform(-1, 1)
    dirac = giagrad.Tensor.empty(2, 3, 4).dirac(groups=2)
    xavier_uniform = giagrad.Tensor.empty(2, 3, 4).xavier_uniform(gain=giagrad.calculate_gain("tanh"))
    xavier_normal = giagrad.Tensor.empty(2, 3, 4).xavier_uniform(gain=giagrad.calculate_gain("conv3d"))
    kaiming_uniform = giagrad.Tensor.empty(4, 5, 6, 7).kaiming_uniform(neg_slope=2, mode='fan_out', nonlinearity='leaky_relu')
    kaiming_normal = giagrad.Tensor.empty(4, 5, 6, 7).kaiming_normal(neg_slope=0.01, mode='fan_out', nonlinearity='leaky_relu')
    sparse = giagrad.Tensor.empty(10, 10).sparse(0.75, 5)
    orthogonal = giagrad.Tensor.empty(10, 10).orthogonal(gain=giagrad.calculate_gain('leaky_relu', neg_slope=13))

    if echo:
        all_ = dict(
            zeros=zeros, 
            ones=ones,
            constant=constant,
            normal=normal,
            uniform=uniform,
            dirac=dirac,
            xavier_uniform=xavier_uniform,
            xavier_normal=xavier_normal,
            kaiming_uniform=kaiming_uniform,
            kaiming_normal=kaiming_normal,
            sparse=sparse,
            orthogonal=orthogonal
        )
        
        for name, t in all_.items():
            print(f"{name}: \n{t.__repr__()}\n")


if __name__ == "__main__":
    test_inits(echo=False)