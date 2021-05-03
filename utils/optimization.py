import torch
from torch.optim import Optimizer


class LARS(Optimizer):
    """
    Implementation of `Large Batch Training of Convolutional Networks, You et al`.
    Code is borrowed from the following:
        https://github.com/cmpark0126/pytorch-LARS
        https://github.com/noahgolmant/pytorch-lars
    """
    def __init__(self,
                 params,
                 lr: float,
                 momentum: float = 0.9,
                 weight_decay: float = 0.,
                 trust_coef: float = 1e-3,
                 dampening: float = 0.,
                 nesterov: bool = False):

        if lr < 0.:
            raise ValueError(f"Invalid `lr` value: {lr}")

        if momentum < 0.:
            raise ValueError(f"Invalid `momentum` value: {momentum}")

        if weight_decay < 0.:
            raise ValueError(f"Invalid `weight_decay` value: {weight_decay}")

        if trust_coef < 0.:
            raise ValueError(f"Invalid `trust_coef` value: {trust_coef}")

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening.")

        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            trust_coef=trust_coef, dampening=dampening, nesterov=nesterov)

        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum     = group['momentum']
            dampening    = group['dampening']
            nesterov     = group['nesterov']
            trust_coef   = group['trust_coef']
            global_lr    = group['lr']

            for p in group['params']:

                if p.grad is None:
                    continue

                d_p = p.grad.data

                p_norm = torch.norm(p.data, p=2)  # weight norm
                d_p_norm = torch.norm(d_p, p=2)   # gradient norm

                local_lr = torch.div(p_norm, d_p_norm + weight_decay * p_norm)
                local_lr.mul_(trust_coef)

                actual_lr = local_lr * global_lr

                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay).mul(actual_lr)

                if momentum != 0:
                    param_state = self.state[p]

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()

                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)

                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-1)

        return loss
