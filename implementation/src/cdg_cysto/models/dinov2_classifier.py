"""DINOv2 classifier wrapper."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from cdg_cysto.models.heads import CDGClassifierHead


class DINOv2Classifier(nn.Module):
    """DINOv2 ViT-S/14 backbone plus CDG-Cysto classifier head."""

    def __init__(
        self,
        num_classes: int,
        hub_model: str = "dinov2_vits14",
        feature_dim: int = 384,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        unfreeze_last_blocks: int = 0,
    ) -> None:
        super().__init__()
        if not pretrained:
            msg = "DINOv2 is expected to use Meta official pretrained weights."
            raise ValueError(msg)

        self.backbone = torch.hub.load("facebookresearch/dinov2", hub_model)
        self.feature_dim = feature_dim
        self.head = CDGClassifierHead(
            in_features=feature_dim,
            hidden_features=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        if freeze_backbone:
            self.freeze_backbone()
        if unfreeze_last_blocks > 0:
            self.unfreeze_last_blocks(unfreeze_last_blocks)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_last_blocks(self, n_blocks: int = 2) -> None:
        if n_blocks <= 0:
            return
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            msg = "DINOv2 backbone does not expose transformer blocks."
            raise AttributeError(msg)
        for block in blocks[-n_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def train_stage1(self) -> None:
        self.freeze_backbone()
        for parameter in self.head.parameters():
            parameter.requires_grad = True

    def train_stage2(self, unfreeze_last_blocks: int = 2) -> None:
        self.freeze_backbone()
        self.unfreeze_last_blocks(unfreeze_last_blocks)
        for parameter in self.head.parameters():
            parameter.requires_grad = True

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            features: Any = self.backbone.forward_features(images)
        else:
            features = self.backbone(images)

        if isinstance(features, dict):
            for key in ("x_norm_clstoken", "cls_token", "pooled"):
                if key in features:
                    features = features[key]
                    break
            else:
                tensor_values = [value for value in features.values() if torch.is_tensor(value)]
                if not tensor_values:
                    msg = "DINOv2 forward_features did not return tensor features."
                    raise RuntimeError(msg)
                features = tensor_values[0]

        if isinstance(features, (tuple, list)):
            features = features[0]

        if features.ndim == 3:
            features = features[:, 0]
        elif features.ndim > 3:
            features = torch.flatten(features, start_dim=1)

        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(images)
        return self.head(features)
