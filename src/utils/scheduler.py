"""MAE-style per-iteration cosine schedule with linear warmup."""
from __future__ import annotations

import math


def lr_at(epoch_f: float, base_lr: float, warmup_epochs: int, total_epochs: int,
          min_lr: float = 0.0) -> float:
    """`epoch_f` is fractional (epoch + step/steps_per_epoch): the schedule advances every
    iteration, not every epoch."""
    if epoch_f < warmup_epochs:
        return base_lr * epoch_f / max(warmup_epochs, 1e-8)
    span = max(total_epochs - warmup_epochs, 1e-8)
    progress = (epoch_f - warmup_epochs) / span
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_lr(optimizer, lr: float) -> float:
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def param_groups_no_decay(model, weight_decay: float):
    """No weight decay on biases, norms or the learned tokens, as is standard for ViTs."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "token" in name or "pos_embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
