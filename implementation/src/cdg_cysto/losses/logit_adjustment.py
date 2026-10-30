"""Logit adjustment loss for long-tailed recognition."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class LogitAdjustmentLoss(nn.Module):
    """Cross entropy after adding tau * log class priors to logits."""

    def __init__(
        self,
        class_priors: Sequence[float],
        *,
        tau: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        priors = torch.as_tensor(list(class_priors), dtype=torch.float32).clamp_min(eps)
        priors = priors / priors.sum()
        self.tau = float(tau)
        self.register_buffer("log_priors", priors.log())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.tau * self.log_priors.to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(adjusted, targets)
