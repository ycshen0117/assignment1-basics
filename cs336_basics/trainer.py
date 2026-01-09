import torch
import torch.nn as nn
import torch.optim as optim
import math
from einops import rearrange, einsum
from collections.abc import Iterable

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute average cross-entropy loss for logits with arbitrary leading batch dims.
    """
    flat_logits = rearrange(logits, "... v -> (...) v")
    flat_targets = rearrange(targets, "... -> (...)")

    max_logits = flat_logits.max(dim=-1, keepdim=True).values
    shifted = flat_logits - max_logits
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))
    target_logits = shifted.gather(-1, flat_targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logits).mean()


class AdamW(optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr * (bias_correction2**0.5) / bias_correction1

                if weight_decay != 0.0:
                    param.add_(param, alpha=-lr * weight_decay)

                denom = exp_avg_sq.sqrt().add_(eps)
                param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
    

def learning_rate_schedule(current_step: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    if current_step < T_w:
        return alpha_max * (current_step / T_w)
    elif current_step <= T_c:
        cos_inner = (math.pi * (current_step - T_w)) / (T_c - T_w)
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(cos_inner))
    else:
        return alpha_min
    

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, epsilon: float = 1e-6) -> None:
    params = [param for param in parameters if param.grad is not None]
    if not params:
        return
    total_norm_sq = torch.zeros((), device=params[0].grad.device, dtype=params[0].grad.dtype)
    for param in params:
        grad = param.grad.detach()
        total_norm_sq.add_(grad.pow(2).sum())
    total_norm = total_norm_sq.sqrt()
    clip_coef = max_l2_norm / (total_norm + epsilon)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    for param in params:
        param.grad.mul_(clip_coef)
