"""Optimizer builders."""

from __future__ import annotations

from torch import nn
from torch.optim import Optimizer


def build_adamw(
    model: nn.Module,
    *,
    learning_rate_head: float = 3e-4,
    learning_rate_backbone: float = 1e-5,
    weight_decay: float = 0.05,
) -> Optimizer:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        msg = "PyTorch is required to build optimizers."
        raise ImportError(msg) from exc

    head_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("head"):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    parameter_groups = []
    if backbone_params:
        parameter_groups.append({"params": backbone_params, "lr": learning_rate_backbone})
    if head_params:
        parameter_groups.append({"params": head_params, "lr": learning_rate_head})
    if not parameter_groups:
        msg = "No trainable parameters found."
        raise ValueError(msg)
    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)
