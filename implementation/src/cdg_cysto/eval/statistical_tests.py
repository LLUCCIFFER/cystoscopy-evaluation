"""Statistical comparison helpers."""

from __future__ import annotations

from math import comb

import numpy as np


def mcnemar_counts(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> dict[str, int]:
    labels = np.asarray(y_true)
    pred_a = np.asarray(y_pred_a)
    pred_b = np.asarray(y_pred_b)
    correct_a = pred_a == labels
    correct_b = pred_b == labels
    return {
        "both_correct": int(np.sum(correct_a & correct_b)),
        "a_correct_b_wrong": int(np.sum(correct_a & ~correct_b)),
        "a_wrong_b_correct": int(np.sum(~correct_a & correct_b)),
        "both_wrong": int(np.sum(~correct_a & ~correct_b)),
    }


def exact_mcnemar_pvalue(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> float:
    counts = mcnemar_counts(y_true, y_pred_a, y_pred_b)
    b = counts["a_correct_b_wrong"]
    c = counts["a_wrong_b_correct"]
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) * (0.5**n) for i in range(k + 1))
    return float(min(1.0, 2.0 * tail))
