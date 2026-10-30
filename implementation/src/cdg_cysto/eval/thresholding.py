"""Threshold selection for binary CDG-Cysto v2 models."""

from __future__ import annotations

import numpy as np


def binary_confusion_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    predictions = (scores >= threshold).astype(int)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {
        "threshold": float(threshold),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
    }


def threshold_grid(y_score: np.ndarray) -> np.ndarray:
    scores = np.asarray(y_score, dtype=float)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    return np.clip(candidates, 0.0, 1.0)


def select_youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    rows = [binary_confusion_at_threshold(y_true, y_score, threshold) for threshold in threshold_grid(y_score)]
    best = max(rows, key=lambda row: row["sensitivity"] + row["specificity"] - 1.0)
    return float(best["threshold"])


def select_min_sensitivity_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    target_sensitivity: float,
) -> float:
    rows = [binary_confusion_at_threshold(y_true, y_score, threshold) for threshold in threshold_grid(y_score)]
    feasible = [row for row in rows if row["sensitivity"] >= target_sensitivity]
    if not feasible:
        return 0.0
    best = max(feasible, key=lambda row: (row["specificity"], row["threshold"]))
    return float(best["threshold"])


def select_min_npv_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    target_npv: float,
) -> float:
    rows = [binary_confusion_at_threshold(y_true, y_score, threshold) for threshold in threshold_grid(y_score)]
    feasible = [row for row in rows if row["npv"] >= target_npv]
    if not feasible:
        return 0.0
    best = max(feasible, key=lambda row: (row["sensitivity"], row["specificity"], row["threshold"]))
    return float(best["threshold"])


def select_standard_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "default_0.5": 0.5,
        "youden": select_youden_threshold(y_true, y_score),
        "sensitivity_90": select_min_sensitivity_threshold(
            y_true,
            y_score,
            target_sensitivity=0.90,
        ),
        "high_npv_95": select_min_npv_threshold(y_true, y_score, target_npv=0.95),
    }
