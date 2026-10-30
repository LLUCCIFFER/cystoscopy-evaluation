"""Attention rollout for ViT-like backbones."""

from __future__ import annotations

import torch


def rollout_attention(attentions: list[torch.Tensor], discard_ratio: float = 0.0) -> torch.Tensor:
    """Compute attention rollout from a list of attention maps.

    Each attention map is expected to have shape `[batch, heads, tokens, tokens]`.
    """

    if not attentions:
        msg = "attentions must contain at least one tensor."
        raise ValueError(msg)

    result = torch.eye(attentions[0].shape[-1], device=attentions[0].device).unsqueeze(0)
    for attention in attentions:
        fused = attention.mean(dim=1)
        if discard_ratio > 0:
            flat = fused.flatten(1)
            _, indices = flat.topk(int(flat.shape[1] * discard_ratio), largest=False)
            flat.scatter_(1, indices, 0)
            fused = flat.view_as(fused)
        identity = torch.eye(fused.shape[-1], device=fused.device).unsqueeze(0)
        augmented = fused + identity
        augmented = augmented / augmented.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        result = torch.bmm(augmented, result)
    return result[:, 0, 1:]
