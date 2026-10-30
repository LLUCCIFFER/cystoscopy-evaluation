from __future__ import annotations

import numpy as np
import pytest

from cdg_cysto.eval.calibration import TemperatureScaler, apply_temperature_np

torch = pytest.importorskip("torch")


def test_apply_temperature_np() -> None:
    logits = np.array([[2.0, 1.0]])
    scaled = apply_temperature_np(logits, 2.0)
    assert np.allclose(scaled, np.array([[1.0, 0.5]]))


def test_temperature_scaler_fits_positive_temperature() -> None:
    logits = torch.tensor([[3.0, 0.1], [0.2, 2.0], [2.0, 0.3], [0.1, 2.5]])
    labels = torch.tensor([0, 1, 0, 1])
    scaler = TemperatureScaler().fit(logits, labels, max_iter=5)
    assert scaler.temperature > 0


def test_temperature_scaling_does_not_change_prediction() -> None:
    logits = np.array([[3.0, 0.1], [0.2, 2.0], [1.0, 0.5]])
    scaled = apply_temperature_np(logits, 3.0)
    assert np.array_equal(logits.argmax(axis=1), scaled.argmax(axis=1))
