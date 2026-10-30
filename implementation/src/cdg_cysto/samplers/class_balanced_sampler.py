"""Class-balanced sampler for imbalanced cystoscopy datasets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def class_counts(labels: Sequence[int], num_classes: int | None = None) -> list[int]:
    values = np.asarray(labels, dtype=int)
    if num_classes is None:
        num_classes = int(values.max()) + 1 if values.size else 0
    return [int((values == index).sum()) for index in range(num_classes)]


def class_weights_from_counts(
    counts: Sequence[int],
    *,
    mode: str = "inverse_frequency",
    beta: float = 0.999,
) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    if np.any(values < 0):
        msg = "class counts must be non-negative"
        raise ValueError(msg)
    safe = np.maximum(values, 1.0)
    if mode == "inverse_frequency":
        weights = 1.0 / safe
    elif mode == "effective_number":
        effective = (1.0 - np.power(beta, safe)) / (1.0 - beta)
        weights = 1.0 / np.maximum(effective, 1e-12)
    else:
        msg = f"Unsupported class-balanced sampler mode: {mode}"
        raise ValueError(msg)
    return weights / weights.sum() * len(weights)


def sample_weights_for_labels(
    labels: Sequence[int],
    *,
    num_classes: int | None = None,
    mode: str = "inverse_frequency",
    beta: float = 0.999,
) -> list[float]:
    counts = class_counts(labels, num_classes)
    weights = class_weights_from_counts(counts, mode=mode, beta=beta)
    return [float(weights[int(label)]) for label in labels]


def build_class_balanced_sampler(
    labels: Sequence[int],
    *,
    num_classes: int | None = None,
    mode: str = "inverse_frequency",
    beta: float = 0.999,
    replacement: bool = True,
):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        msg = "PyTorch is required for class-balanced sampling."
        raise ImportError(msg) from exc

    weights = sample_weights_for_labels(labels, num_classes=num_classes, mode=mode, beta=beta)
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=replacement,
    )
