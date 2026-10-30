from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cdg_cysto.losses.class_balanced import (  # noqa: E402
    ClassBalancedFocalLoss,
    effective_number_weights,
)


def test_effective_number_weights_minor_class_is_larger() -> None:
    weights = effective_number_weights(torch.tensor([100, 10]), beta=0.99)
    assert weights[1] > weights[0]


def test_class_balanced_focal_loss_backward() -> None:
    logits = torch.tensor([[2.0, 0.2], [0.1, 1.5]], requires_grad=True)
    targets = torch.tensor([0, 1])
    loss = ClassBalancedFocalLoss([10, 2])(logits, targets)
    loss.backward()
    assert logits.grad is not None
