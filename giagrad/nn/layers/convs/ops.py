from giagrad.tensor import Tensor
from giagrad.nn.layers.convs.params import ConvParams 
import giagrad.nn.layers.convs.utils as utils  
import numpy as np
from numpy.typing import NDArray
from typing import Optional

def convolve(
        x: NDArray, w: NDArray, params: ConvParams, backward: Optional[ConvParams] = None
    ) -> NDArray:

        # output tensor shape 
        output_shape = utils.conv_output_shape(x.shape, params)  

        if not np.all(output_shape > 0):
            raise ValueError(
                    "Stride, dilation, padding and kernel dimensions are incompatible"
                    )  

        # apply padding if at least one non-zero entry
        if params.needs_padding:
            x = np.pad(
                x, 
                pad_width=params.axis_pad(x), 
                mode=params.padding_mode, 
                **params.padding_kwargs
            ) 
        
        # make a view of x ready for tensordot
        sliding_view = utils.sliding_filter_view(array=x, params=params)
        
        axes = [
            [-(axis+1) for axis in range(params.conv_dims)]
        ]*2 # convolve the last conv_dims dimensions of w and sliding_view

        conv_out = np.tensordot(
            w, 
            sliding_view,
            axes=axes
        ) 
        
        if not conv_out.flags['C_CONTIGUOUS']:
            return  np.ascontiguousarray(conv_out)
        return conv_out
    