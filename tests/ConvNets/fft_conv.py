# https://github.com/fkodom/fft-conv-pytorch/blob/master/fft_conv_pytorch/fft_conv.py
import sys
sys.path.append('../../')
from giagrad.nn.layers.utils import conv_output_shape
import numpy as np
np.random.seed(1234)

BATCH = 1
KERNEL_SIZE = (2, 2)
INPUT_SIZE = (5, 5)
IN_CHANNELS = 2 
OUT_CHANNELS = 2
STRIDE = (1, 1)
DILATION = (1, 1)
PADDING = (1, 3)
GROUPS = 1

SIGNAL_SHAPE = (BATCH,) + (IN_CHANNELS,) + INPUT_SIZE
KERNEL_SHAPE = (OUT_CHANNELS,) + (IN_CHANNELS,) + (KERNEL_SIZE)

# init tensors
signal = np.random.randint(4, size=SIGNAL_SHAPE).astype(np.float64)
kernel = np.random.randint(2, size=KERNEL_SHAPE).astype(np.float64)

print(f"{signal = }")
print(f"{kernel = }")

print(f"{signal.shape = }")
print(f"{kernel.shape = }")

output_shape = conv_output_shape(
    signal.shape,
    KERNEL_SIZE,
    STRIDE, 
    DILATION,
    PADDING
)

# signal.ndim = 4
n = signal.ndim - 2

# internal dilation offsets
offset = np.zeros((1, 1, *DILATION))
offset[(..., *((0,) * n))] = 1.0


# correct the kenrel by cutting off unwantd dilation trailing zeros
cutoff = tuple(slice(None, -d+1 if d != 1 else None) for d in DILATION)

# pad the kernel internally according to the dilation parameters
kernel = np.kron(kernel, offset)[(...,) + cutoff]

print(f"dilated kernel {kernel}")

# pad the input signal & kernel tensors
signal_padding = [(0,0)]*2 + [(p,p) for p in PADDING]
signal = np.pad(signal, signal_padding)


if signal.shape[-1] % 2 != 0:
    signal_ = np.pad(signal, [(0,0)]*(signal.ndim-1) + [(0, 1)])
else:
    signal_ = signal

print(f"check if signal padding has even size\n{signal_ = }")

kernel_padding = [(0,0)]*2 + [
        tuple(pad for pad in [0, signal_.shape[i] - kernel.shape[i]])
        for i in range(2, signal_.ndim)
    ]

print(F"{kernel_padding = }")

padded_kernel = np.pad(kernel, kernel_padding)

print(f"{padded_kernel = }")

signal_fr = np.fft.rfftn(signal_, axes=tuple(range(2, signal.ndim)))
kernel_fr = np.fft.rfftn(padded_kernel, axes=tuple(range(2, 4)))

kimag = np.imag(kernel_fr) 
kimag *= -1

print(f"{signal.shape = }")
print(f"{kernel.shape = }")
print(f"{signal_fr.shape = }")
print(f"{kernel_fr.shape = }")

### COMPLEX MULTIPLICATION
a = signal_fr.reshape(
    (signal_.shape[0], GROUPS, -1, *signal_fr.shape[2:])
)
b = kernel_fr.reshape(
    (GROUPS, -1, *kernel_fr.shape[1:])
)

print(f"{a.shape = }")
print(f"{b.shape = }")

a = np.moveaxis(a, 2, -1)
a = np.expand_dims(a, -2)

b = np.moveaxis(b, (1, 2), (-1, -2))

print(f"{a.shape = }")
print(f"{b.shape = }")

real = np.real(a) @ np.real(b) - np.imag(a) @ np.imag(b)
imag = np.imag(a) @ np.real(b) + np.real(a) @ np.imag(b)
real = np.squeeze(np.moveaxis(real, -1, 2), -1)
imag = np.squeeze(np.moveaxis(imag, -1, 2), -1)

c = np.zeros(real.shape, dtype=np.complex64)
np.real(c)[:] = real
np.imag(c)[:] = imag

print(f"{c.shape = }")

output_fr = c.reshape((c.shape[0], -1, *c.shape[3:]))
output = np.fft.irfftn(output_fr, axes=tuple(range(2, signal.ndim)))

crop_slices = (
    (...,) 
    + tuple(
        slice(0, (signal.shape[i] - kernel.shape[i] + 1), STRIDE[i - 2])
        for i in range(2, signal.ndim)
    )
)
output = output[crop_slices]
print("\n\n### OUTPUT ###")
print(f"output shape {output_shape}")
print(f"{output.shape = }")
print(f"{output = }")
# print(f"{signal_ = }")
# print(f"{padded_kernel = }")
# print(f"{signal_fr.shape = }")
# print(f"{kernel_fr.shape = }")
