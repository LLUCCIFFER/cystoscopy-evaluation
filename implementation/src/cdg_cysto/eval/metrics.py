"""Classification, calibration, and ranking metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def ensure_probabilities(y_score: np.ndarray) -> np.ndarray:
    score = np.asarray(y_score, dtype=float)
    if score.ndim != 2:
        msg = "y_score must be a 2D array of logits or probabilities."
        raise ValueError(msg)
    row_sums = score.sum(axis=1)
    if np.all(score >= 0) and np.allclose(row_sums, 1.0, atol=1e-5):
        return score
    return softmax(score)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    probabilities = ensure_probabilities(y_prob)
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies = predictions == np.asarray(y_true)

    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for start, end in zip(bins[:-1], bins[1:], strict=False):
        in_bin = (confidences > start) & (confidences <= end)
        if not np.any(in_bin):
            continue
        prop = float(np.mean(in_bin))
        ece += prop * abs(float(np.mean(accuracies[in_bin])) - float(np.mean(confidences[in_bin])))
    return float(ece)


def brier_score_multiclass(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int | None = None) -> float:
    probabilities = ensure_probabilities(y_prob)
    if num_classes is None:
        num_classes = probabilities.shape[1]
    one_hot = np.eye(num_classes)[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def per_class_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    total = matrix.sum()
    result: dict[str, float] = {}

    for index, class_name in enumerate(class_names):
        tp = matrix[index, index]
        fn = matrix[index, :].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = total - tp - fn - fp
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        key = str(class_name).lower()
        result[f"sensitivity_{key}"] = float(sensitivity)
        result[f"specificity_{key}"] = float(specificity)

    result["sensitivity_macro"] = float(
        np.nanmean([result[f"sensitivity_{str(name).lower()}"] for name in class_names])
    )
    result["specificity_macro"] = float(
        np.nanmean([result[f"specificity_{str(name).lower()}"] for name in class_names])
    )
    return result


def auroc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    probabilities = ensure_probabilities(y_prob)
    labels = np.asarray(y_true, dtype=int)
    num_classes = probabilities.shape[1]
    try:
        if num_classes == 2:
            return float(roc_auc_score(labels, probabilities[:, 1]))
        return float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def auprc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    probabilities = ensure_probabilities(y_prob)
    labels = np.asarray(y_true, dtype=int)
    num_classes = probabilities.shape[1]
    try:
        if num_classes == 2:
            return float(average_precision_score(labels, probabilities[:, 1]))
        binarized = label_binarize(labels, classes=list(range(num_classes)))
        return float(average_precision_score(binarized, probabilities, average="macro"))
    except ValueError:
        return float("nan")


def classification_curves(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    probabilities = ensure_probabilities(y_prob)
    labels = np.asarray(y_true, dtype=int)
    if probabilities.shape[1] != 2:
        msg = "Curves are currently emitted for binary tasks only."
        raise ValueError(msg)
    fpr, tpr, roc_thresholds = roc_curve(labels, probabilities[:, 1])
    precision, recall, pr_thresholds = precision_recall_curve(labels, probabilities[:, 1])
    return {
        "roc": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
        "pr": {"precision": precision, "recall": recall, "thresholds": pr_thresholds},
    }


def compute_classification_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int] | None = None,
    y_score: np.ndarray | list[list[float]] | None = None,
    class_names: list[str] | tuple[str, ...] | None = None,
    n_bins: int = 15,
) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    if class_names is None:
        num_classes = int(labels.max()) + 1
        class_names = [str(index) for index in range(num_classes)]
    else:
        num_classes = len(class_names)

    probabilities = ensure_probabilities(np.asarray(y_score, dtype=float)) if y_score is not None else None
    predictions = (
        np.asarray(y_pred, dtype=int)
        if y_pred is not None
        else probabilities.argmax(axis=1)
        if probabilities is not None
        else None
    )
    if predictions is None:
        msg = "Either y_pred or y_score must be provided."
        raise ValueError(msg)

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }
    metrics.update(per_class_sensitivity_specificity(labels, predictions, list(class_names)))

    if probabilities is not None:
        metrics["auroc"] = auroc_score(labels, probabilities)
        metrics["auprc"] = auprc_score(labels, probabilities)
        metrics["ece"] = expected_calibration_error(labels, probabilities, n_bins=n_bins)
        metrics["brier"] = brier_score_multiclass(labels, probabilities, num_classes)
        try:
            metrics["nll"] = float(log_loss(labels, probabilities, labels=list(range(num_classes))))
        except ValueError:
            metrics["nll"] = float("nan")

    return metrics


def build_predictions_frame(
    sample_ids: list[str],
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[list[float]],
) -> pd.DataFrame:
    """Create the required predictions_test.csv table."""

    probabilities = ensure_probabilities(np.asarray(y_prob, dtype=float))
    labels = np.asarray(y_true, dtype=int)
    predictions = probabilities.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "y_true": labels,
            "y_pred": predictions,
        }
    )
    for class_index in range(probabilities.shape[1]):
        frame[f"prob_{class_index}"] = probabilities[:, class_index]
    return frame


def save_confusion_matrix_plot(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    class_names: list[str] | tuple[str, ...],
    path: str | Path,
) -> None:
    """Save a confusion matrix plot."""

    import matplotlib.pyplot as plt

    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(labels, class_names, rotation=30, ha="right")
    ax.set_yticks(labels, class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for row in labels:
        for col in labels:
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)


def save_roc_curve_plot(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[list[float]],
    path: str | Path,
) -> None:
    """Save ROC curve plot for binary or one-vs-rest multiclass tasks."""

    import matplotlib.pyplot as plt

    labels = np.asarray(y_true, dtype=int)
    probabilities = ensure_probabilities(np.asarray(y_prob, dtype=float))
    num_classes = probabilities.shape[1]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    if num_classes == 2:
        try:
            fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
            ax.plot(fpr, tpr, label=f"AUROC {roc_auc_score(labels, probabilities[:, 1]):.3f}")
        except ValueError:
            ax.plot([0, 1], [0, 1], label="AUROC n/a")
    else:
        binarized = label_binarize(labels, classes=list(range(num_classes)))
        for class_index in range(num_classes):
            if len(np.unique(binarized[:, class_index])) < 2:
                continue
            fpr, tpr, _ = roc_curve(binarized[:, class_index], probabilities[:, class_index])
            ax.plot(fpr, tpr, alpha=0.7, label=f"class {class_index}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)


def save_pr_curve_plot(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[list[float]],
    path: str | Path,
) -> None:
    """Save precision-recall curve plot."""

    import matplotlib.pyplot as plt

    labels = np.asarray(y_true, dtype=int)
    probabilities = ensure_probabilities(np.asarray(y_prob, dtype=float))
    num_classes = probabilities.shape[1]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    if num_classes == 2:
        try:
            precision, recall, _ = precision_recall_curve(labels, probabilities[:, 1])
            ap = average_precision_score(labels, probabilities[:, 1])
            ax.plot(recall, precision, label=f"AUPRC {ap:.3f}")
        except ValueError:
            ax.plot([0, 1], [1, 0], label="AUPRC n/a")
    else:
        binarized = label_binarize(labels, classes=list(range(num_classes)))
        for class_index in range(num_classes):
            if len(np.unique(binarized[:, class_index])) < 2:
                continue
            precision, recall, _ = precision_recall_curve(
                binarized[:, class_index], probabilities[:, class_index]
            )
            ax.plot(recall, precision, alpha=0.7, label=f"class {class_index}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)


def save_reliability_diagram(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[list[float]],
    path: str | Path,
    *,
    n_bins: int = 10,
) -> None:
    """Save a reliability diagram based on maximum class confidence."""

    import matplotlib.pyplot as plt

    labels = np.asarray(y_true, dtype=int)
    probabilities = ensure_probabilities(np.asarray(y_prob, dtype=float))
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies = predictions == labels
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    observed = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for index, (start, end) in enumerate(zip(bins[:-1], bins[1:], strict=False)):
        mask = (confidences > start) & (confidences <= end)
        counts[index] = mask.sum()
        observed[index] = np.mean(accuracies[mask]) if np.any(mask) else np.nan

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.bar(centers, np.nan_to_num(observed), width=1 / n_bins, alpha=0.75, edgecolor="black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"ECE {expected_calibration_error(labels, probabilities, n_bins=n_bins):.3f}")
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
