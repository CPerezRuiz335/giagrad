from giagrad.tensor import Tensor
import numpy as np
from typing import List 
from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, params: List[Tensor]):
        self.params = [t for t in params if t.requires_grad]
        self.ite = 1

    @abstractmethod
    def step(self):
        """Performs a single optimization step/epoch (parameter update)."""
        raise NotImplementedError(f"step not implemented for {type(self)}")

    def zero_grad(self):
        """Sets the gradients of all optimized tensors to zero."""
        for p in self.params: 
            p.grad *= 0 # np.zeros_like(p.grad, dtype=np.float32) 


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    
    Based on `PyTorch SGD`_.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{13mm} \:\mu \text{ (momentum)}, \:\tau \text{ (dampening)},
            \:\textit{ nesterov,}\:\textit{ maximize}                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}\textbf{if} \: \mu \neq 0                                               \\
            &\hspace{10mm}\textbf{if} \: t > 1                                                   \\
            &\hspace{15mm} \textbf{b}_t \leftarrow \mu \textbf{b}_{t-1} + (1-\tau) g_t           \\
            &\hspace{10mm}\textbf{else}                                                          \\
            &\hspace{15mm} \textbf{b}_t \leftarrow g_t                                           \\
            &\hspace{10mm}\textbf{if} \: \textit{nesterov}                                       \\
            &\hspace{15mm} g_t \leftarrow g_{t} + \mu \textbf{b}_t                             \\
            &\hspace{10mm}\textbf{else}                                                   \\[-1.ex]
            &\hspace{15mm} g_t  \leftarrow  \textbf{b}_t                                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}                                          \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} + \gamma g_t                   \\[-1.ex]
            &\hspace{5mm}\textbf{else}                                                    \\[-1.ex]
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma g_t                   \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`_.
    
    .. _PyTorch SGD: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    .. _On the importance of initialization and momentum in deep learning: http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    
    Attributes
    ----------
    params: iterable of Tensor
        Iterable of parameters to optimize.
    lr: float, default: 0.001
        Learning rate.
    momentum: float, default: 0 
        Momentum factor.
    weight_decay: float, default: 0
        Weight decay (L2 penalty).
    dampening: float, default: 0 
        Dampening for momentum.
    nesterov: bool, default: False 
        Enables Nesterov momentum.
    maximize: bool, default: False
        Maximize the params based on the objective, instead of minimizing.
    
    Examples
    --------
    >>> optimizer = giagrad.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> model.zero_grad()
    >>> loss_fn(model(input), target).backward()
    >>> optimizer.step()
    """
    def __init__(
            self, 
            params, 
            lr=1e-3, 
            momentum=0.,
            weight_decay=0.,
            dampening=0., 
            nesterov=False, 
            maximize=False
        ):
        assert not nesterov or (momentum != 0  and dampening == 0), \
            "Nesterov momentum requires a momentum and zero dampening"
        super().__init__(params)
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
        self.dampening, self.nesterov, self.maximize = dampening, nesterov, maximize
        self.b = [np.zeros(t.shape) for t in self.params]

    # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    def step(self):
        for t, b in zip(self.params, self.b):
            g = t.grad

            if self.weight_decay != 0:
                g += self.weight_decay * t.data

            if self.momentum != 0:
                if self.ite > 1:
                    b[:] = self.momentum * b + (1 - self.dampening) * g
                else:
                    b[:] = g 

                if self.nesterov:
                    g += self.momentum * b 
                else:
                    g[:] = b 

            if self.maximize:
                t.data += self.lr * g 
            else:                
                t.data -= self.lr * g

        self.ite += 1