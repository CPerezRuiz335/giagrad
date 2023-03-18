# giagrad
Deep learning framework made by and for students.

Like [micrograd](https://youtu.be/VMj-3S1tku0). More like [tinygrad](https://github.com/geohot/tinygrad) but with the spirit of 
[numpy_ml](https://numpy-ml.readthedocs.io/en/latest/) but more PyTorch-ish. See [micrograd](https://youtu.be/VMj-3S1tku0) to understand.

# TODO
- Documentation
- Add more optimizers, layers, etc

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

# PROBLEMS
- PyTorch CrossEntropyLoss it's like giagrad's CrossEntropyLoss but doing mean(dim=0).sum() (/giagrad/nn/loss/CrossEntropyLoss line 36). This may cause a significant difference in the loss value for reduction mean, but the gradients will only differ by a constant value k, where k is equal to the number of observations in the input data. The problem is that doing mean(dim=0).sum() like PyTorch, makes the MLP not learn.