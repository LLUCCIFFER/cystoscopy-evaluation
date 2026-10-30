"""Scheduler builders."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_cosine_scheduler(optimizer: Optimizer, total_steps: int, warmup_steps: int = 0) -> LambdaLR:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        msg = "PyTorch is required to build schedulers."
        raise ImportError(msg) from exc

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
