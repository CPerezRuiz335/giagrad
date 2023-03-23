# giagrad
Deep learning framework made by and for students.

Like [micrograd](https://youtu.be/VMj-3S1tku0). More like [tinygrad](https://github.com/geohot/tinygrad) but with the spirit of 
[numpy_ml](https://numpy-ml.readthedocs.io/en/latest/) but more PyTorch-ish. See [micrograd](https://youtu.be/VMj-3S1tku0) to understand.

# TODO
- Documentation
- Create init like pytorch but not in nn
https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_normal_
to be able to Tensor.empty(*shape).xavier_uniform(**kwargs) and modify in place
- Add more optimizers, layers, etc
- Start dropout layers after previous point
- dim or axis parameter?

# PROBLEMS
- optimization and speed VS simplicity and self-explained code for newbies
- lack of contributions/community :man_shrugging:

# GOAL
- code almost everything popular in AI even transformers

# FUTURE BRANCHES/IDEAS
- nn/layers
- optimizers
- loss functions

# COULD BE NICE
> "The guy who knows about computers is the last person you want to have creating documentation for people who don't understand computers." ~ Adam Osborne
- documentation
- visualization 
