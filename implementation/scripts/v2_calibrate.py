"""Fit temperature scaling on CystoDS calib logits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdg_cysto.eval.calibration import TemperatureScaler, calibration_metrics  # noqa: E402
from cdg_cysto.eval.metrics import ensure_probabilities  # noqa: E402
from cdg_cysto.utils.io import write_json, write_table  # noqa: E402


def _logit_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith("logit_")],
        key=lambda value: int(value.split("_")[1]),
    )


def _prediction_frame(source: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    predictions = source[["sample_id", "y_true"]].copy()
    for index in range(probabilities.shape[1]):
        predictions[f"prob_{index}"] = probabilities[:, index]
    predictions["y_pred"] = probabilities.argmax(axis=1)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-logits", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.calib_logits)
    columns = _logit_columns(frame)
    logits = frame[columns].to_numpy(dtype=float)
    labels = frame["y_true"].to_numpy(dtype=int)
    scaler = TemperatureScaler().fit(logits, labels)
    calibrated_logits = scaler.transform(logits)
    if (logits.argmax(axis=1) != calibrated_logits.argmax(axis=1)).any():
        msg = "Temperature scaling changed argmax predictions."
        raise RuntimeError(msg)

    before = calibration_metrics(labels, logits)
    after = calibration_metrics(labels, calibrated_logits)
    rows = [
        {"metric": key, "before": before[key], "after": after[key], "delta_after_minus_before": after[key] - before[key]}
        for key in ("ece", "brier", "nll")
    ]
    write_table(pd.DataFrame(rows), output / "calibration_metrics_before_after.csv")
    write_json({"temperature": scaler.temperature}, output / "temperature.json")
    calibrated = frame.copy()
    for index, column in enumerate(columns):
        calibrated[column] = calibrated_logits[:, index]
    write_table(calibrated, output / "logits_calibrated.csv")
    calibrated_probabilities = ensure_probabilities(calibrated_logits)
    write_table(_prediction_frame(frame, calibrated_probabilities), output / "predictions_calibrated.csv")
    print(f"temperature={scaler.temperature}")


if __name__ == "__main__":
    main()
