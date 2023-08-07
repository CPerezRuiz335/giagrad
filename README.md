# giagrad
Deep learning framework made by and for students.

Like [micrograd](https://youtu.be/VMj-3S1tku0). More like [tinygrad](https://github.com/geohot/tinygrad) but with the spirit of 
[numpy_ml](https://numpy-ml.readthedocs.io/en/latest/) but more PyTorch-ish. See [micrograd](https://youtu.be/VMj-3S1tku0) to understand.

# TODO
- Pooling layers.

# GOAL
- code almost everything popular in AI even transformers

# PROBLEMS
- optimization and speed VS simplicity and self-explained code

# OTHER
giagrad/tensor may not seem concise with all those docstrings, but try to remove them
with this regex pattern ((\s+r"""*)(.|\n)+?("""))|((\s+"""*)(.|\n)+?(""")) and
you'll barely see 300 lines of code in every file.
