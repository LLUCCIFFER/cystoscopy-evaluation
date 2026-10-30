"""Feature-attention classifiers for CDG-Cysto v3 lesion-aware experiments."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

TIMM_BACKBONES = {
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "convnext_tiny": "convnext_tiny",
}


class TinyAttentionCNN(nn.Module):
    """Small attention-capable CNN for fast tests and smoke configs."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, images: torch.Tensor, *, return_attention: bool = False) -> Any:
        features = self.features(images)
        logits = self.classifier(self.pool(features).flatten(1))
        if not return_attention:
            return logits
        attention = features.abs().mean(dim=1, keepdim=True)
        return {"logits": logits, "attention": attention}


class FeatureAttentionClassifier(nn.Module):
    """Timm features-only backbone with a simple spatial attention map."""

    def __init__(
        self,
        *,
        name: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        normalized = name.lower()
        if normalized == "tiny_cnn":
            self.impl: nn.Module | None = TinyAttentionCNN(num_classes)
            self.backbone = None
            self.pool = None
            self.classifier = None
            self.feature_channels = 32
            return
        if normalized not in TIMM_BACKBONES:
            msg = f"Unsupported lesion-aware backbone: {name}"
            raise ValueError(msg)
        try:
            import timm
        except ImportError as exc:  # pragma: no cover
            msg = "timm is required for v3 lesion-aware classifiers."
            raise ImportError(msg) from exc
        self.impl = None
        self.backbone = timm.create_model(
            TIMM_BACKBONES[normalized],
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),
        )
        self.feature_channels = int(self.backbone.feature_info.channels()[-1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_channels),
            nn.Dropout(dropout),
            nn.Linear(self.feature_channels, num_classes),
        )

    def _feature_nchw(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            msg = f"Expected 4D feature map, got shape {tuple(features.shape)}"
            raise ValueError(msg)
        if features.shape[1] == self.feature_channels:
            return features
        if features.shape[-1] == self.feature_channels:
            return features.permute(0, 3, 1, 2).contiguous()
        msg = f"Could not infer feature channel layout from shape {tuple(features.shape)}"
        raise ValueError(msg)

    def forward(self, images: torch.Tensor, *, return_attention: bool = False) -> Any:
        if self.impl is not None:
            return self.impl(images, return_attention=return_attention)
        if self.backbone is None or self.pool is None or self.classifier is None:
            msg = "FeatureAttentionClassifier is not initialized."
            raise RuntimeError(msg)
        features = self._feature_nchw(self.backbone(images)[-1])
        pooled = self.pool(features).flatten(1)
        logits = self.classifier(pooled)
        if not return_attention:
            return logits
        attention = features.abs().mean(dim=1, keepdim=True)
        return {"logits": logits, "attention": attention}
