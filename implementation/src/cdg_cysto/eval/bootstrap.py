"""Bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    labels = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    if len(labels) != len(predictions):
        msg = "y_true and y_pred must have the same length."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(labels), size=len(labels))
        values.append(metric_fn(labels[indices], predictions[indices]))

    alpha = 1.0 - confidence
    return {
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(values, alpha / 2)),
        "upper": float(np.quantile(values, 1 - alpha / 2)),
    }
