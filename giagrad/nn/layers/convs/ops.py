from giagrad.tensor import Tensor
from giagrad.nn.layers.convs.params import ConvParams 
import giagrad.nn.layers.convs.utils as utils  
import numpy as np
from numpy.typing import NDArray
from typing import Optional

def convolve(
        xt: Tensor, wt: Tensor, params: ConvParams, backward: Optional[ConvParams] = None
    ) -> NDArray:

    