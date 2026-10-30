"""Temperature scaling for confidence calibration."""

from __future__ import annotations

import numpy as np

from cdg_cysto.eval.metrics import (
    brier_score_multiclass,
    ensure_probabilities,
    expected_calibration_error,
    softmax,
)


def apply_temperature_np(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        msg = "temperature must be positive"
        raise ValueError(msg)
    return np.asarray(logits, dtype=float) / temperature


class TemperatureScaler:
    """Single-parameter temperature scaling fitted on validation logits."""

    def __init__(self, init_temperature: float = 1.0) -> None:
        if init_temperature <= 0:
            msg = "init_temperature must be positive"
            raise ValueError(msg)
        self.init_temperature = init_temperature
        self.temperature_: float | None = None

    @property
    def temperature(self) -> float:
        return self.init_temperature if self.temperature_ is None else self.temperature_

    def fit(
        self,
        logits,
        labels,
        *,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> TemperatureScaler:
        try:
            import torch
            from torch.nn import functional as F
        except ImportError as exc:  # pragma: no cover
            msg = "PyTorch is required to fit temperature scaling."
            raise ImportError(msg) from exc

        logits_tensor = torch.as_tensor(logits, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.long)
        log_temperature = torch.nn.Parameter(torch.log(torch.tensor(float(self.init_temperature))))
        optimizer = torch.optim.LBFGS([log_temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature).clamp_min(1e-6)
            loss = F.cross_entropy(logits_tensor / temperature, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature_ = float(torch.exp(log_temperature).detach().cpu().item())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return apply_temperature_np(np.asarray(logits, dtype=float), self.temperature)

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return softmax(self.transform(logits))


def calibration_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    *,
    n_bins: int = 15,
) -> dict[str, float]:
    """Return ECE, Brier score, and NLL for logits or probabilities."""

    from sklearn.metrics import log_loss

    probabilities = ensure_probabilities(logits)
    num_classes = probabilities.shape[1]
    try:
        nll = float(log_loss(labels, probabilities, labels=list(range(num_classes))))
    except ValueError:
        nll = float("nan")
    return {
        "ece": expected_calibration_error(labels, probabilities, n_bins=n_bins),
        "brier": brier_score_multiclass(labels, probabilities, num_classes=num_classes),
        "nll": nll,
    }


def fit_temperature_and_report(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    *,
    n_bins: int = 15,
) -> tuple[TemperatureScaler, dict[str, float]]:
    """Fit temperature on validation logits and report test calibration before/after."""

    scaler = TemperatureScaler().fit(val_logits, val_labels)
    raw = calibration_metrics(test_labels, test_logits, n_bins=n_bins)
    calibrated_logits = scaler.transform(test_logits)
    calibrated = calibration_metrics(test_labels, calibrated_logits, n_bins=n_bins)
    if not np.array_equal(np.argmax(test_logits, axis=1), np.argmax(calibrated_logits, axis=1)):
        msg = "Temperature scaling changed predicted classes, which should not happen."
        raise RuntimeError(msg)
    report = {"temperature": scaler.temperature}
    report.update({f"raw_{key}": value for key, value in raw.items()})
    report.update({f"calibrated_{key}": value for key, value in calibrated.items()})
    return scaler, report
