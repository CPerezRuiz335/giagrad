from giagrad.tensor import Tensor
from giagrad.nn.layers.convs.params import ConvParams 
import giagrad.nn.layers.convs.utils as utils  
import numpy as np
from numpy.typing import NDArray
from typing import Optional

def convolve(
        xt: Tensor, wt: Tensor, params: ConvParams, backward: Optional[ConvParams] = None
    ) -> NDArray:

    x, w = xt.data, wt.data

    # output tensor shape 
    output_shape = utils.output_shape(x.shape, params)  

    if not np.all(output_shape > 0):
        raise ValueError(
                "Stride, dilation, padding and kernel dimensions are incompatible"
                )  

    # apply padding if at least one non-zero entry
    if params.needs_padding:
        x = np.pad(
            x, params.axis_pad, 
            mode=params.padding_mode, 
            **params.padding_kwargs
        ) 
    
    # trims x if filter does not cover entire sample 
    # if backward pass, derivatives are not computed 
    # with respect of uncovered data
    x = utils.trimm_uneven_stride(
        array=x, 
        params=params if backward is None else backward
    )

    # make a view of x ready for tensordot
    sliding_view = utils.sliding_filter_view(array=x, params=params), 
    
    axes = [
        [-(axis+1) for axis in range(params.conv_dims)]
    ]*2 # convolve the last conv_dims dimensions of w and sliding_view

    conv_out = np.tensordot(
        w, 
        sliding_view,
        axes=axes
    ) 
    if not params.online_learning:
        # (C_out, N, W0, ...) -> (N, C_out, W0, ...)
        conv_out = np.swapaxes(conv_out, 0, 1)
    
    if not conv_out.flags['C_CONTIGUOUS']:
        conv_out = np.ascontiguousarray(conv_out)

    return conv_out