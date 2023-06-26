from typing import List, Tuple

import numpy as np

from giagrad.tensor import Tensor
from giagrad.optim.optimizer import Optimizer


class Adam(Optimizer):
    r"""Implements Adam algorithm. 
    
    This algorithm calculates the exponential moving average of gradients and square gradients. 
    And the parameters of β1 and β2 are used to control the decay rates of these moving averages.
    
    Based on `PyTorch Adam`_.

    .. math::

       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)}          \\
            &\hspace{13mm}      \lambda \text{ (weight decay)},  \: \textit{amsgrad},
                \:\textit{maximize}                                                              \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0\leftarrow 0 \text{ (second moment)},\: \widehat{v_0}^{max}\leftarrow 0       \\
            &\rule{110mm}{0.4pt}                                                                 \\[-1.ex]
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}
    
    .. _PyTorch Adam: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam

    Attributes
    ----------
    params: iterable of Tensor
        Iterable of parameters to optimize.
    lr: float, default: 0.001
        Learning rate.
    betas: Tuple[float,float], default: (0.9,0.999)
        Betas.
    eps: float, default: 1e-8
        Epsilon value.
    weight_decay: float, default: 0
        Weight decay (L2 penalty).
    maximize: bool, default: False
        Maximize the params based on the objective, instead of minimizing.
    amsgrad: bool, default: False
        Option to use the AMSGrad variant of this algorithm.

    Examples
    --------
    >>> optimizer = giagrad.optim.Adam(model.parameters(), lr=0.1, amsgrad=True)
    >>> model.zero_grad()
    >>> loss_fn(model(input), target).backward()
    >>> optimizer.step()

    .. _On the Convergence of Adam and Beyond: https://openreview.net/pdf?id=ryQu7f-RZ
    """

    def __init__(
            self, 
            params: List[Tensor],
            lr:float = 1e-3,
            betas:Tuple[float,float] = (0.9,0.999),
            eps:float = 1e-8,
            weight_decay:float = 0.,
            maximize:bool = False,
            amsgrad:bool = False
        ):
        super().__init__(params)
        self.lr, self.eps, self.weight_decay = lr, eps, weight_decay
        self.beta1, self.beta2 = betas
        self.maximize, self.amsgrad = maximize, amsgrad
        self.m = [np.zeros(p.shape) for p in self.params]
        self.v = [np.zeros(p.shape) for p in self.params]
        self.vmax_ = [np.zeros(p.shape) for p in self.params]

    def step(self):
        for t, m, v, vmax_ in zip(self.params, self.m, self.v, self.vmax_):
            if self.maximize:
                g = -t.grad.copy()
            else:
                g = t.grad.copy()

            if self.weight_decay != 0:
                g += self.weight_decay * t.data

            m[:] = np.add(self.beta1 * m, (1-self.beta1) * g)
            v[:] = np.add(self.beta2 * v, (1-self.beta2) * g**2)
            m_ = m / (1-self.beta1**self.ite)
            v_ = v / (1-self.beta2**self.ite)

            if self.amsgrad:
                vmax_[:] = np.maximum(vmax_, v_)
                t.data -= (self.lr * m_) / (np.sqrt(vmax_+self.eps))
            else:
                t.data -= (self.lr * m_) / (np.sqrt(v_+self.eps))

        self.ite += 1