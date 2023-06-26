import numpy as np

from giagrad.tensor import Tensor
from giagrad.optim.optimizer import Optimizer

class Adadelta(Optimizer):
    r"""Implements Adadelta algorithm.
    
    Based on `PyTorch Adadelta`_.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)},
                \: f(\theta) \text{ (objective)}, \: \rho \text{ (decay)},
                \: \lambda \text{ (weight decay)}                                                \\
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{initialize} :  v_0  \leftarrow 0 \: \text{ (square avg)},
                \: u_0 \leftarrow 0 \: \text{ (accumulate variables)}                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm} v_t      \leftarrow v_{t-1} \rho + g^2_t (1 - \rho)                    \\
            &\hspace{5mm}\Delta x_t    \leftarrow   \frac{\sqrt{u_{t-1} +
                \epsilon }}{ \sqrt{v_t + \epsilon}  }g_t \hspace{21mm}                           \\
            &\hspace{5mm} u_t  \leftarrow   u_{t-1}  \rho +
                 \Delta x^2_t  (1 - \rho)                                                        \\
            &\hspace{5mm}\theta_t      \leftarrow   \theta_{t-1} - \gamma  \Delta x_t            \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\
       \end{aligned}

    For further details regarding the algorithm we refer to 
    `ADADELTA: An Adaptive Learning Rate Method 
    <https://arxiv.org/abs/1212.5701>`_.
   
    Parameters
    ----------
    params: iterable of Tensor
        Iterable parameters to optimize.
    rho: float, default: 0.9 
        Coefficient used for computing a running average of squared gradients.
    eps: float, default: 1e-6
        Term added to the denominator to improve numerical stability.
    lr: float, default: 1.0
        Coefficient that scale delta before it is applied to the parameters.
    weight_decay: float, default: 0
        Weight decay (L2 penalty).
    maximize: bool, default: False
        Maximize the params based on the objective, instead of minimizing.

    Examples
    --------
    >>> optimizer = giagrad.optim.Adadelta(model.parameters())
    >>> model.zero_grad()
    >>> loss_fn(model(input), target).backward()
    >>> optimizer.step()

    .. _PyTorch Adadelta: https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html
    """

    def __init__(
            self, 
            params,
            lr=1.,
            rho=0.,
            eps=1e-6,
            weight_decay=0.,
            maximize=False
        ):
        super().__init__(params)
        self.lr, self.eps, self.rho = lr, eps, rho
        self.maximize, self.weight_decay = maximize, weight_decay
        self.u = [np.zeros(p.shape) for p in self.params]
        self.v = [np.zeros(p.shape) for p in self.params]

    def step(self):
        for t, u, v in zip(self.params, self.u, self.v):
            if self.maximize:
                g = -t.grad.copy()
            else:
                g = t.grad.copy()

            if self.weight_decay != 0:
                g += self.weight_decay * t.data

            v[:] = v * self.rho + g**2 * (1-self.rho)
            x = (np.sqrt(u + self.eps) / np.sqrt(v + self.eps)) * g
            u[:] = u*self.rho + x**2 * (1-self.rho)
            t.data -= self.lr * x

        self.ite += 1
