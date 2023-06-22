import numpy as np

from giagrad.nn.layers.utils import dilate, transpose, conv_output_shape


shape = (7, 3, 5, 5)
kernel = (3, 9, 2, 4)
input_conv = conv_output_shape(
	(10, 12),
	kernel[-2:],
	stride=(2,2),
	dilation=(1,1),
	padding=(0,)
)
t = np.arange(np.prod(shape)).reshape(shape)
w = np.arange(np.prod(kernel)).reshape(kernel)
trans = transpose(
	t,
	w,
	stride=(2,2),
	dilation=(1,1),
)


print(f"{input_conv = }")
print(f"{trans.shape = }")