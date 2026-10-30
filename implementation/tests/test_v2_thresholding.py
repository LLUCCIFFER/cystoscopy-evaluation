from __future__ import annotations

import numpy as np

from cdg_cysto.eval.thresholding import binary_confusion_at_threshold, select_standard_thresholds


def test_v2_threshold_selection_outputs_required_names() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.8, 0.9])
    thresholds = select_standard_thresholds(y_true, y_score)
    assert set(thresholds) == {"default_0.5", "youden", "sensitivity_90", "high_npv_95"}
    metrics = binary_confusion_at_threshold(y_true, y_score, thresholds["default_0.5"])
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
