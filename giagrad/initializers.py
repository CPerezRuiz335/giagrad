"""https://pytorch.org/docs/stable/_modules/torch/nn/init.html"""
import numpy as np
import warnings
import math

def _calculate_fan_in_and_fan_out(tensor):
    if tensor.ndim < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1 if tensor.ndim > 2 else math.prod(tensor.shape[2:])
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def _calculate_correct_fan(tensor, mode: str):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
       raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def calculate_gain(nonlinearity, neg_slope=None):
    r"""
    Returns the recommended gain value for a spcefic nonlinear fuction. 

    Some initializers are derived from specific nonlinear functions
    such as Kaiming uniform or Kaiming normal through PReLU definition
    and have a recommended gain associated.

    The values are as follow:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative_slope}^2}}`
    SELU              :math:`\frac{3}{4}`
    ================= ====================================================

    .. warning::
        In order to implement `Self-Normalizing Neural Networks`_ ,
        you should use ``nonlinearity='linear'`` instead of ``nonlinearity='selu'``.
        This gives the initial weights a variance of ``1 / N``,
        which is necessary to induce a stable fixed point in the forward pass.
        In contrast, the default gain for ``SELU`` sacrifices the normalisation
        effect for more stable gradient flow in rectangular layers.

    Parameters
    ----------

        nonlinearity: str
            the non-linear method name
        neg_slope: Scalar
            optional negative slope constant for Leaky ReLU 

    Examples
    --------
        >>> giagrad.calculate_gain('leaky_relu', 2)  # leaky_relu with negative_slope=0.2                                                    
        0.6324555320336759

    .. _Self-Normalizing Neural Networks: https://papers.nips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if neg_slope is None:
            negative_slope = 0.01
        elif not isinstance(neg_slope, bool) and isinstance(neg_slope, int) or isinstance(neg_slope, float):
            # True/False are instances of int, hence check above
            negative_slope = neg_slope
        else:
            raise ValueError("negative_slope {} not a valid number".format(neg_slope))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")

def normal(tensor, mu, sigma):
    tensor.data = np.random.default_rng().normal(loc=mu, scale=sigma, size=tensor.shape).astype(tensor.dtype)

def uniform(tensor, a, b):
    tensor.data = np.random.default_rng().uniform(low=a, high=b, size=tensor.shape).astype(tensor.dtype)

def dirac(tensor, groups=1):
    dimensions = tensor.ndim
    if dimensions not in [3, 4, 5]:
        raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")

    sizes = tensor.shape

    if sizes[0] % groups != 0:
        raise ValueError('dim 0 must be divisible by groups')

    out_chans_per_grp = sizes[0] // groups
    min_dim = min(out_chans_per_grp, sizes[1])

    tensor.zeros()

    for g in range(groups):
        for d in range(min_dim):
            if dimensions == 3:  # Temporal convolution
                tensor.data[g * out_chans_per_grp + d, d, sizes[2] // 2] = 1
            elif dimensions == 4:  # Spatial convolution
                tensor.data[g * out_chans_per_grp + d, d, sizes[2] // 2,
                       tensor.size(3) // 2] = 1
            else:  # Volumetric convolution
                tensor.data[g * out_chans_per_grp + d, d, sizes[2] // 2,
                       tensor.size(3) // 2, tensor.size(4) // 2] = 1

def xavier_uniform(tensor, gain=1):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    tensor.uniform(-a, a)

def xavier_normal(tensor, gain=1):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    tensor.normal(0, std)

def kaiming_uniform(tensor, neg_slope, mode, nonlinearity):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, neg_slope)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    tensor.uniform(-bound, bound)

def kaiming_normal(tensor, neg_slope, mode, nonlinearity):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, neg_slope)
    std = gain / math.sqrt(fan)
    tensor.normal(0, std)

def orthogonal(tensor, gain):
    if tensor.ndim < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")
    rows = tensor.shape[0]
    cols = tensor.size // rows
    flattened = np.random.default_rng().standard_normal(size=(rows, cols))

    if rows < cols:
        flattened = flattened.T

    # Compute the qr factorization
    q, r = np.linalg.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = np.diag(r, 0)
    ph = np.sign(d)
    q *= ph

    if rows < cols:
        q = q.T

    tensor.data.reshape(q.shape) * gain

def sparse(tensor, sparsity, std):
    if tensor.ndim != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    rows, cols = tensor.shape
    num_zeros = int(math.ceil(sparsity * rows))

    tensor.normal(0, std)
    for col_idx in range(cols):
        row_indices = np.random.permutation(rows)
        zero_indices = row_indices[:num_zeros]
        tensor.data[zero_indices, col_idx] = 0