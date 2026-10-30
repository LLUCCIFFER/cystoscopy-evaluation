from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cdg_cysto.losses.balanced_softmax import BalancedSoftmaxLoss  # noqa: E402
from cdg_cysto.losses.logit_adjustment import LogitAdjustmentLoss  # noqa: E402
from cdg_cysto.samplers.class_balanced_sampler import sample_weights_for_labels  # noqa: E402


def test_v2_sampler_weights_minority_higher() -> None:
    weights = sample_weights_for_labels([0, 0, 0, 1], mode="inverse_frequency")
    assert weights[-1] > weights[0]


def test_v2_balanced_softmax_and_logit_adjustment_loss() -> None:
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    loss1 = BalancedSoftmaxLoss([10, 2])(logits, targets)
    loss2 = LogitAdjustmentLoss([0.8, 0.2])(logits, targets)
    assert torch.isfinite(loss1)
    assert torch.isfinite(loss2)
