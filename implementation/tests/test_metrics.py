from __future__ import annotations

import numpy as np

from cdg_cysto.eval.metrics import (
    brier_score_multiclass,
    build_predictions_frame,
    compute_classification_metrics,
    expected_calibration_error,
    softmax,
)


def test_softmax_rows_sum_to_one() -> None:
    probs = softmax(np.array([[1.0, 2.0], [3.0, 1.0]]))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_compute_classification_metrics_binary() -> None:
    y_true = np.array([0, 1, 1, 0])
    logits = np.array([[3.0, 0.0], [0.1, 2.0], [0.2, 1.5], [1.0, 0.2]])
    metrics = compute_classification_metrics(y_true, y_score=logits, class_names=["non_cancer", "cancer"])
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["auroc"] == 1.0
    assert "ece" in metrics


def test_calibration_metrics_are_finite() -> None:
    y_true = np.array([0, 1, 2])
    probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.1, 0.7]])
    assert np.isfinite(expected_calibration_error(y_true, probs))
    assert np.isfinite(brier_score_multiclass(y_true, probs))


def test_predictions_frame_contains_required_columns() -> None:
    frame = build_predictions_frame(
        ["a", "b"],
        [0, 1],
        np.array([[0.8, 0.2], [0.1, 0.9]]),
    )
    assert frame.columns.tolist() == ["sample_id", "y_true", "y_pred", "prob_0", "prob_1"]
