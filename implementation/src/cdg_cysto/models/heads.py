"""Classifier heads used by CDG-Cysto models."""

from __future__ import annotations

import torch
from torch import nn


class CDGClassifierHead(nn.Module):
    """LayerNorm + MLP head used for DINOv2 features."""

    def __init__(
        self,
        in_features: int = 384,
        hidden_features: int = 256,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
