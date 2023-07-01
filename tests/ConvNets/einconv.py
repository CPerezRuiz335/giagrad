import numpy as np
from timeit import timeit

# 2D conv

BATCH = 124
IN_CHANNELS = 32
OUTPUT_SIZE = (28, 28)

OUT_CHANNELS = 32
KERNEL_SIZE = (10, 10)

winow_view = np.random.rand(
	BATCH, *OUTPUT_SIZE, IN_CHANNELS, *KERNEL_SIZE
)

w = np.random.rand(
	OUT_CHANNELS, IN_CHANNELS, *KERNEL_SIZE
)

## einsum
def einsum_vanilla():
	np.einsum("B OS C kl, c C kl -> BcOS", winow_view, w)

def einsum_true():
	np.einsum(
		"B OS C kl, c C kl -> BcOS", 
		winow_view, w,
		optimize=True
	)

def einsum_greedy():
	np.einsum(
		"B OS C kl, c C kl -> BcOS", 
		winow_view, w,
		optimize='greedy'
	)

def einsum_optim():
	np.einsum(
		"B OS C kl, c C kl -> BcOS", 
		winow_view, w,
		optimize='optimal'
	)

path = np.einsum_path(
	"B OS C kl, c C kl -> BcOS",
	winow_view, w,
	optimize='optimal'
)[0]
def einsum_path():
	np.einsum(
		"B OS C kl, c C kl -> BcOS", 
		winow_view, w,
		optimize=path
	)

def tensordot():
	a = np.tensordot(winow_view, w, axes=[[-1, -2, -3]]*2)
	b = np.rollaxis(a, -1, 1)

if __name__ == "__main__":
	# print(f"{timeit(einsum_vanilla, number=10) = }")
	print(f"{timeit(einsum_true, number=10) = }")
	print(f"{timeit(einsum_greedy, number=10) = }")
	print(f"{timeit(einsum_optim, number=10) = }")
	print(f"{timeit(einsum_path, number=10) = }")
	print(f"{timeit(tensordot, number=10) = }")
