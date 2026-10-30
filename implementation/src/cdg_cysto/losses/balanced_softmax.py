"""Balanced Softmax loss."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class BalancedSoftmaxLoss(nn.Module):
    """Cross entropy after adding log class counts to logits."""

    def __init__(self, class_counts: Sequence[int | float], eps: float = 1e-12) -> None:
        super().__init__()
        counts = torch.as_tensor(list(class_counts), dtype=torch.float32).clamp_min(eps)
        self.register_buffer("log_counts", counts.log())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.log_counts.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(adjusted, targets)
