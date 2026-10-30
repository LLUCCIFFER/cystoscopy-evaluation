"""Model factory for fixed comparison set M0-M3."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from cdg_cysto.models.dinov2_classifier import DINOv2Classifier

TIMM_BACKBONES = {
    "efficientnet_b3": "efficientnet_b3",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "convnext_tiny": "convnext_tiny",
    "resnet50": "resnet50",
}


class TinyCNN(nn.Module):
    """Small CNN used only for v2 smoke tests."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, images: Any) -> Any:
        features = self.features(images).flatten(1)
        return self.classifier(features)


def create_model(config: Mapping[str, Any]) -> nn.Module:
    name = str(config.get("name", "")).lower()
    source = str(config.get("source", "")).lower()
    num_classes = int(config.get("num_classes", 4))

    if name == "tiny_cnn":
        return TinyCNN(num_classes=num_classes)

    if source == "timm" or name in TIMM_BACKBONES:
        try:
            import timm
        except ImportError as exc:  # pragma: no cover
            msg = "timm is required for EfficientNet-B3, Swin-Tiny, and ConvNeXt-Tiny baselines."
            raise ImportError(msg) from exc

        backbone = str(config.get("backbone") or TIMM_BACKBONES.get(name))
        return timm.create_model(
            backbone,
            pretrained=bool(config.get("pretrained", True)),
            num_classes=num_classes,
        )

    if name in {"cdg_cysto_dinov2", "dinov2_linear"} or "dinov2" in name:
        classifier = config.get("classifier", {}) or {}
        training = config.get("training", {}) or {}
        return DINOv2Classifier(
            num_classes=num_classes,
            hub_model=str(config.get("backbone", "dinov2_vits14")),
            feature_dim=int(config.get("feature_dim", 384)),
            hidden_dim=int(classifier.get("hidden_dim", 256)),
            dropout=float(classifier.get("dropout", 0.2)),
            pretrained=bool(config.get("pretrained", True)),
            freeze_backbone=bool(config.get("freeze_backbone", False))
            or bool(training.get("stage1_freeze_backbone", False)),
            unfreeze_last_blocks=int(config.get("unfreeze_last_blocks", 0)),
        )

    msg = f"Unsupported model config: {config}"
    raise ValueError(msg)


def configure_training_stage(model: nn.Module, stage: int, unfreeze_last_blocks: int = 2) -> None:
    if hasattr(model, "train_stage1") and stage == 1:
        model.train_stage1()
        return
    if hasattr(model, "train_stage2") and stage == 2:
        model.train_stage2(unfreeze_last_blocks=unfreeze_last_blocks)
        return

    for parameter in model.parameters():
        parameter.requires_grad = True
