"""Frozen CystoDS internal test or EBTC external validation for CDG-Cysto v2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdg_cysto.eval.calibration import apply_temperature_np  # noqa: E402
from cdg_cysto.eval.metrics import (  # noqa: E402
    compute_classification_metrics,
    ensure_probabilities,
)
from cdg_cysto.eval.thresholding import binary_confusion_at_threshold  # noqa: E402
from cdg_cysto.models.factory import create_model  # noqa: E402
from cdg_cysto.train.engine_v2 import V2MetadataDataset, _build_transform, collate_v2  # noqa: E402
from cdg_cysto.utils.checkpoint import load_checkpoint  # noqa: E402
from cdg_cysto.utils.device import get_device  # noqa: E402
from cdg_cysto.utils.io import read_json, write_table  # noqa: E402


def _collect(checkpoint_path: Path, metadata: pd.DataFrame, label_column: str) -> tuple[np.ndarray, list[str]]:
    import torch

    device = get_device(prefer_cuda=True)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    task = config.get("task", {}) or {}
    config = dict(config)
    config["task"] = dict(task)
    config["task"]["label_column"] = label_column
    model_config = dict(config.get("model", {}) or {})
    model_config["num_classes"] = int(task.get("num_classes", 2))
    model = create_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = V2MetadataDataset(metadata, label_column=label_column, transform=_build_transform(config, train=False))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_v2)
    logits_all: list[np.ndarray] = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            logits_all.append(model(batch["image"].to(device)).detach().cpu().numpy())
            sample_ids.extend(batch["sample_id"])
    return np.concatenate(logits_all), sample_ids


def _write_logits(sample_ids: list[str], labels: np.ndarray, logits: np.ndarray, path: Path) -> None:
    frame = pd.DataFrame({"sample_id": sample_ids, "y_true": labels})
    for index in range(logits.shape[1]):
        frame[f"logit_{index}"] = logits[:, index]
    write_table(frame, path)


def _predictions(sample_ids: list[str], labels: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"sample_id": sample_ids, "y_true": labels})
    for index in range(probabilities.shape[1]):
        frame[f"prob_{index}"] = probabilities[:, index]
    frame["y_pred"] = probabilities.argmax(axis=1)
    return frame


def _metrics_by_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for name, threshold in thresholds.items():
        y_pred = (probabilities[:, 1] >= float(threshold)).astype(int)
        metrics = compute_classification_metrics(
            labels,
            y_pred=y_pred,
            y_score=probabilities,
            class_names=["non_cancer", "cancer"],
        )
        row = {"threshold_name": name, "threshold": float(threshold)}
        row.update(metrics)
        row.update(binary_confusion_at_threshold(labels, probabilities[:, 1], float(threshold)))
        rows.append(row)
    return pd.DataFrame(rows)


def _metrics_by_ebtc_class(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    working = metadata.copy().reset_index(drop=True)
    rows = []
    scores = probabilities[:, 1]
    for threshold_name, threshold in thresholds.items():
        working["y_pred"] = (scores >= threshold).astype(int)
        for label, group in working.groupby("label_original"):
            indices = group.index.to_numpy()
            true_positive_class = int(group["label_cancer_binary"].iloc[0]) == 1
            predicted_cancer = working.loc[indices, "y_pred"].to_numpy(dtype=int)
            predicted_cancer_rate = float(predicted_cancer.mean())
            rows.append(
                {
                    "threshold_name": threshold_name,
                    "threshold": float(threshold),
                    "ebtc_class": label,
                    "n": len(group),
                    "sensitivity": predicted_cancer_rate if true_positive_class else float("nan"),
                    "specificity": (1.0 - predicted_cancer_rate)
                    if not true_positive_class
                    else float("nan"),
                    "false_positive_rate": predicted_cancer_rate
                    if not true_positive_class
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--temperature-json", required=True)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="label_cancer_binary")
    parser.add_argument("--external-ebtc", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.split_csv)
    metadata = metadata[metadata["is_valid"].astype(bool)].reset_index(drop=True)
    labels = metadata[args.label_column].astype(int).to_numpy()
    logits, sample_ids = _collect(Path(args.checkpoint), metadata, args.label_column)
    temperature = float(read_json(args.temperature_json)["temperature"])
    thresholds = {key: float(value) for key, value in read_json(args.thresholds_json).items()}
    calibrated_logits = apply_temperature_np(logits, temperature)
    probabilities = ensure_probabilities(calibrated_logits)
    predictions = _predictions(sample_ids, labels, probabilities)
    write_table(predictions, output / "predictions.csv")
    _write_logits(sample_ids, labels, calibrated_logits, output / "logits.csv")
    threshold_metrics = _metrics_by_threshold(labels, probabilities, thresholds)
    write_table(threshold_metrics, output / "metrics_by_threshold.csv")
    if args.external_ebtc:
        write_table(
            _metrics_by_ebtc_class(metadata, probabilities, thresholds),
            output / "metrics_by_ebtc_class.csv",
        )
    print(f"Wrote validation outputs to {output}")


if __name__ == "__main__":
    main()
