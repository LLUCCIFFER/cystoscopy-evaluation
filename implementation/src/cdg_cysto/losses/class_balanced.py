"""Class-balanced weighting and focal loss."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def effective_number_weights(
    samples_per_class: torch.Tensor,
    beta: float = 0.999,
    normalize: bool = True,
) -> torch.Tensor:
    counts = samples_per_class.float().clamp_min(1)
    effective_num = 1.0 - torch.pow(torch.tensor(beta, device=counts.device), counts)
    weights = (1.0 - beta) / effective_num
    if normalize:
        weights = weights / weights.sum() * len(weights)
    return weights


class ClassBalancedFocalLoss(nn.Module):
    """Class-Balanced Focal Loss for imbalanced classification."""

    def __init__(
        self,
        samples_per_class: list[int] | torch.Tensor,
        beta: float = 0.999,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            msg = "reduction must be one of: mean, sum, none"
            raise ValueError(msg)
        counts = torch.as_tensor(samples_per_class, dtype=torch.float32)
        self.register_buffer("class_weights", effective_number_weights(counts, beta))
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets = targets.long()

        ce_loss = F.nll_loss(log_probs, targets, weight=self.class_weights, reduction="none")
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        focal_factor = torch.pow(1.0 - pt, self.gamma)
        loss = focal_factor * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
