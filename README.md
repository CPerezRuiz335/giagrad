# giagrad
Deep learning framework made by and for students.

Like [micrograd](https://youtu.be/VMj-3S1tku0). More like [tinygrad](https://github.com/geohot/tinygrad) but with the spirit of 
[numpy_ml](https://numpy-ml.readthedocs.io/en/latest/) but more PyTorch-ish. See [micrograd](https://youtu.be/VMj-3S1tku0) to understand.

# PERFORMANCE conclusions
sgemm not good enough, plain numpy well optimizer
convolutionals with toeplitz impossible and fft too slow
transposed convolution with direct convolution far better
try without tensordort: memory leakage
tensordot is the best even with matrix multiplication and reshaping is far far better

# TODO
- Improve linear layer with scipy BLAS dgemm routines
- TEST TOEPLITZ MATRIX WITH CONV AND BLAS ROUTINES FROM SCIPY
- Add more optimizers, layers, etc
- Start convolution layers (see tests)
- Add layer padding, update documentation of new shapeops
- retry adding github actions for documentation

# GOAL
- code almost everything popular in AI even transformers

# PROBLEMS
- optimization and speed VS simplicity and self-explained code

# OTHER
giagrad/tensor may not seem concise with all those docstrings, but try to remove them
with this regex pattern ((\s+r"""*)(.|\n)+?("""))|((\s+"""*)(.|\n)+?(""")) and
you'll barely see 300 lines of code in every file.
