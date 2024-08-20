from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        """
            在 PyTorch 的 Optimizer 类中，param_groups 是一个列表，列表中的每个元素都是一个字典，代表了一组参数（parameters）的优化配置。
            每个字典通常包含以下键值对：

            params: 一个列表，包含一组需要优化的参数（通常是模型的权重和偏置）。
            lr: 学习率（learning rate）。
            weight_decay: 权重衰减（weight decay）。
            momentum: 动量（momentum）。
            dampening: 阻尼（dampening）。
            nesterov: 是否使用 Nesterov 动量（Nesterov momentum）。
            param_groups 的主要目的是允许优化器同时优化模型中不同部分的参数，使用不同的优化配置。例如，你可以定义两个 param_groups，一个用于优化模型的权重，另一个用于优化模型的偏置。

            在 AdamW 优化器的实现中，param_groups 列表中的每个元素都是一个字典，包含以下键值对：

            params: 一组需要优化的参数。
            lr: 学习率。
            betas: 两个超参数，用于计算梯度的第一和第二个矩。
            eps: 一个小值，用于避免除以零。
            weight_decay: 权重衰减。
            correct_bias: 是否使用偏差修正。
        """

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]


                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                state['step'] += 1

                # # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                # https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for
                # Question One: Why we use bias correction in AdamW?
                # Answer One: If we don't use it, the effect of Gt0 will persist a long time in our V and Momentum things
                # Answer Two: If we don't use it, the value of M1 is 0.1 Gt0, is so small compared to the correct gradient
                # Answer Three: It's a method to balance the value of M0, making it closer to the average of gradient
                # Answer Four: Make it larger, so it has more possibilities to jump out of local minimum

                # Question Two: Why we don't use Gt0 to initial value of M0?
                # Answer One: It will make the effect of Gt0, persist a long time
                
                correct_bias = group["correct_bias"]

                # Update first and second moments
                state['exp_avg'].mul_(beta1).add_(grad.mul(1 - beta1))
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad * (1 - beta2))

                # Apply bias correction
                if correct_bias:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                else: 
                    bias_correction1 = 1.0
                    bias_correction2 = 1.0

                exp_avg_corrected = state['exp_avg'] / bias_correction1

                exp_avg_sq_corrected = state['exp_avg_sq'] / bias_correction2
             
                # Update parameters
                p.data.addcdiv_(-alpha * exp_avg_corrected, exp_avg_sq_corrected.sqrt().add(eps) )

                # Apply weight decay
                if weight_decay != 0:
                    p.data.add_(-weight_decay * alpha * p.data)

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # To complete this implementation:
                # 1. Update the first and second moments of the gradients.
                # 2. Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3. Update parameters (p.data).
                # 4. Apply weight decay after the main gradient-based updates.
                # Refer to the default project handout for more details.


        return loss
