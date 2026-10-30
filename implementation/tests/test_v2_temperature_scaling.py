from __future__ import annotations

import numpy as np

from cdg_cysto.eval.calibration import TemperatureScaler


def test_v2_temperature_scaling_preserves_argmax() -> None:
    logits = np.array([[3.0, 1.0], [0.5, 2.0], [2.0, 0.0], [0.0, 2.0]])
    labels = np.array([0, 1, 0, 1])
    scaler = TemperatureScaler().fit(logits, labels)
    calibrated = scaler.transform(logits)
    assert np.array_equal(logits.argmax(axis=1), calibrated.argmax(axis=1))
