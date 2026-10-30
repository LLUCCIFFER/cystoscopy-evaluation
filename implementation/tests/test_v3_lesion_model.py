from __future__ import annotations

import torch

from cdg_cysto.models.lesion_aware import FeatureAttentionClassifier


def test_v3_tiny_attention_classifier_forward() -> None:
    model = FeatureAttentionClassifier(name="tiny_cnn", num_classes=3, pretrained=False)
    output = model(torch.randn(2, 3, 96, 96), return_attention=True)
    assert output["logits"].shape == (2, 3)
    assert output["attention"].shape[0] == 2
    assert output["attention"].shape[1] == 1
