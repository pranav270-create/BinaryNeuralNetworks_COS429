import numpy as np
import pandas as pd
import math
import torch
import matplotlib.pyplot as plt
import os
import matplotlib
import copy
from collections import OrderedDict


class Adam_Metaplastic(torch.optim.Optimizer):
    """ For this custom Adam optimizer with metaplasticity, we modified the Pytorch 
    source code for the Adam optimizer (https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam)
    and used the algorithm explained in Laborieux, et al. to create the metaplastic weight update.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), meta=1.3, eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        self.defaults = dict(lr=lr, betas=betas, eps=eps, meta=meta,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam_Metaplastic, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_Metaplastic, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

        for i, param in enumerate(params_with_grad):

            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if group['weight_decay'] != 0:
                grad = grad.add(param, alpha=group['weight_decay'])

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

            denom = (exp_avg_sq.sqrt() /
                     math.sqrt(bias_correction2)).add_(group['eps'])

            step_size = group['lr'] / bias_correction1

            # BINARIZED NN STEP

            # Computes binary weight (sign of the current weight)
            binary_weight = torch.sign(param.data)
            # Computes condition where binary weight has same sign as the Adam update
            metaplastic_condition = (torch.mul(binary_weight, exp_avg) > 0.0)

            if param.dim() == 1:
                # Only the case where the param is the bias and not a weight
                param.data.addcdiv_(exp_avg, denom, value=-step_size)
            else:
                # Compute metaplastic weight update --> Use f_meta to increase the weight more
                meta_avg = torch.mul(torch.ones_like(param.data) - torch.pow(torch.tanh(defaults['meta']*torch.abs(param.data)), 2), exp_avg)
                # Metaplastic Update --> Wherever binary weight has same sign as Adam, use metaplatic weight update. Otherwise,
                # use normal Adam weight update (exp_avg)
                param.data.addcdiv_(torch.where(metaplastic_condition, meta_avg, exp_avg), denom, value=-step_size)

        return loss
