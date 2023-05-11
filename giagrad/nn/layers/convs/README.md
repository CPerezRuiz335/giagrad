Quick notation for some docstrings, e.g. 3D convolution 
with multiple channels with batch training (an image of C_in 
channels now is understood as a 3 dimensional image with Depth,
Height and Width):

    N: number of observations in batch
    C_in: number of input channels
    D_in: input depth size
    H_in: input height size
    W_in: input width size

For higher dimensions same pattern applies, let's say
fourth dimension is G_in:

    N: number of observations in batch
    C_in: number of input channels
    G_in: input number of 3D arrays of size (D_in, H_in, W_in)
    D_in: input depth size
    H_in: input height size
    W_in: input width size

For kernels, the dimensions are:
    
    C_out: number of channels each observations has after applying 
            that convolution, i.e. number of kernels/filters to 
            be applied.
    C_in: every kernel/filter has the same channels as the input 
            image (if no grouped convolution).
    kH: kernel height 
    kW: kernel widht

Likewise, for higher order convolutions such as 3D ones, every filter
has also kD (filter depth).  

    kernel_shape is the shape of the entire filter or kernel.
    kernel_size is the shape of each single filter without channels.
    conv_dims equals the length of kernel_size.

For the sake of modularity, convolutions and transposed convolutions 
are separated in different files, in ops.py the main functionality is
defined in two functions: convolve and transpose. These functions are
used in forward or backpropagation of \_ConvND and \_TransposeND.     

In addition, ConvParams is defined to simplify the definition of every function
involved in this module, because convolutions require a bunch parameters.
