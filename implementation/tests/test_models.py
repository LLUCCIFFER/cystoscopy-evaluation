from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cdg_cysto.models.dinov2_classifier import DINOv2Classifier  # noqa: E402
from cdg_cysto.models.factory import create_model  # noqa: E402


class DummyDINO(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(384, 384) for _ in range(4)])

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = images.shape[0]
        return {"x_norm_clstoken": torch.ones(batch_size, 384, device=images.device)}


def test_dinov2_classifier_output_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load(repo: str, model: str) -> DummyDINO:
        assert repo == "facebookresearch/dinov2"
        assert model == "dinov2_vits14"
        return DummyDINO()

    monkeypatch.setattr(torch.hub, "load", fake_load)
    model = DINOv2Classifier(num_classes=4, freeze_backbone=True)
    logits = model(torch.zeros(2, 3, 224, 224))
    assert logits.shape == (2, 4)
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())


def test_factory_supports_convnext_tiny_without_pretrained() -> None:
    timm = pytest.importorskip("timm")
    model = create_model(
        {
            "name": "convnext_tiny",
            "source": "timm",
            "pretrained": False,
            "num_classes": 3,
        }
    )
    assert model is not None
    assert timm is not None
    model.eval()
    logits = model(torch.zeros(1, 3, 224, 224))
    assert logits.shape == (1, 3)
