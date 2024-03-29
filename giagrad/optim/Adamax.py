import numpy as np

from giagrad.tensor import Tensor
from giagrad.optim.optimizer import Optimizer

class Adamax(Optimizer):
    r"""Implements Adamax algorithm (a variant of Adam based on infinity norm).
    
    Based on `PyTorch Adamax`_.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \beta_1, \: \beta_2 \text{ (betas)}, \: 
                \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{13mm} \:\epsilon \text{ (epsilon)}                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{initialize} : m_0 \leftarrow 0 \: \text{(first moment)}, \: u_0 \leftarrow 0 \: 
            \text{(infinity norm)}                                                               \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\                         \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t \leftarrow \beta_1 m_{t-1} + (1-\beta_1) g_t                        \\
            &\hspace{5mm}u_t \leftarrow \max(\beta_2 u_{t-1}, |g_t| + \epsilon)                  \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} - \frac{(1-\beta_1^t)}{u_t} \gamma m_t \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}
    
    .. _PyTorch Adamax: https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html#torch.optim.Adamax

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

    Examples
    --------
    >>> optimizer = giagrad.optim.Adamax(model.parameters())
    >>> model.zero_grad()
    >>> loss_fn(model(input), target).backward()
    >>> optimizer.step()
    """

    def __init__(
            self, 
            params,
            lr=1e-3,
            betas=(0.9,0.999),
            eps=1e-8,
            weight_decay=0.,
            maximize=False
        ):
        super().__init__(params)
        self.lr, self.eps, self.weight_decay = lr, eps, weight_decay
        self.beta1, self.beta2 = betas
        self.maximize = maximize
        self.m = [np.zeros(p.shape) for p in self.params]
        self.u = [np.zeros(p.shape) for p in self.params]

    def step(self):
        for t, m, u in zip(self.params, self.m, self.u):
            g = t.grad.copy()

            if self.weight_decay != 0:
                g += self.weight_decay * t.data

            m[:] = self.beta1 * m + (1-self.beta1) * g
            u[:] = np.maximum(self.beta2 * u, np.abs(g + self.eps))
            t.data -= (self.lr * m) / ((1 - self.beta1**self.ite) * u)
            
        self.ite += 1