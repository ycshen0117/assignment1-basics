import torch
from einops import rearrange
import math
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


def compute_entropy_chunked(logits:torch.Tensor, chunk_size:int=128) -> torch.Tensor:
    """Memory-efficient implementation of `compute_entropy`."""
    num_chunks = (logits.shape[1] + chunk_size - 1) // chunk_size
    entropy_chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, logits.shape[1])
        chunk_logits = logits[:, start_idx:end_idx, :]
        
        # Use the numerically stable method for torch.bfloat16, do not use logsumexp
        chunk_probs = chunk_logits.softmax(dim=-1)
        chunk_log_probs = chunk_logits.log_softmax(dim=-1)
        chunk_entropy = -(chunk_probs * chunk_log_probs).sum(dim=-1)
        entropy_chunks.append(chunk_entropy)
    return torch.cat(entropy_chunks, dim=1)


def learning_rate_schedule(current_step: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    if current_step < T_w:
        return alpha_max * (current_step / T_w)
    elif current_step <= T_c:
        cos_inner = (math.pi * (current_step - T_w)) / (T_c - T_w)
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(cos_inner))
    else:
        return alpha_min
    

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, epsilon: float = 1e-6) -> torch.Tensor | None:
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
    return total_norm