from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cdg_cysto.models.factory import create_model  # noqa: E402


def test_v2_tiny_cnn_forward() -> None:
    model = create_model({"name": "tiny_cnn", "num_classes": 2})
    logits = model(torch.zeros(2, 3, 32, 32))
    assert logits.shape == (2, 2)


def test_v2_resnet50_factory_without_pretrained() -> None:
    pytest.importorskip("timm")
    model = create_model({"name": "resnet50", "source": "timm", "pretrained": False, "num_classes": 2})
    logits = model(torch.zeros(1, 3, 64, 64))
    assert logits.shape == (1, 2)
