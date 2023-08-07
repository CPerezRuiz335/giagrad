import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from giagrad.nn.layers.utils import sliding_filter_view
import numpy as np
np.random.seed(1234)

BATCH = 2
IN_CHANNELS = 3
INPUT_SIZE = (5, 5)
KERNEL_SIZE = (2, 2)
STRIDE = (1, 1)
DILATION = (1, 1)
conv_dims = len(KERNEL_SIZE)

t = np.random.randint(
	5, 
	size=(BATCH, IN_CHANNELS, *INPUT_SIZE)
)
print(f"{t = }")

original_window_view = sliding_filter_view(t, KERNEL_SIZE, STRIDE, DILATION)
original_shape = original_window_view.shape
print(f"{original_window_view.shape = }")

window_view = original_window_view.reshape((*original_window_view.shape[:-conv_dims], -1))
print(f"{window_view.shape = }")

indices = np.argmax(window_view, axis=-1)
print(f"{indices = }")
offset = np.arange(np.prod(indices.shape)).reshape(indices.shape) 
# print(f"{offset = }")
offset *= np.prod(KERNEL_SIZE)
print(f"{offset = }")

indices += offset
print(f"{indices = }")
print(f"{window_view.flatten() = }")


print(f"{indices.shape = }")
print(f"{offset.shape = }")

flat_indices = np.unravel_index(indices.flatten(), original_shape)
print(f"{flat_indices = }")
print(f"{original_window_view[flat_indices]}")

# offset = np.prod(window_view.shape[:-1])
# print(f"{offset = }")

# print(f"{indices = }")
# print(f"{indices.shape = }")


# indices_offset = (indices + offset).flatten()
# print(f"{indices_offset = }")

print(f"{t = }")
np.add.at(original_window_view, flat_indices, 100)
print(f"{t = }")




