"""EBTC official-test-split sensitivity analysis from frozen predictions."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdg_cysto.eval.metrics import compute_classification_metrics  # noqa: E402
from cdg_cysto.eval.thresholding import binary_confusion_at_threshold  # noqa: E402
from cdg_cysto.utils.io import write_table  # noqa: E402

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "sensitivity_cancer",
    "specificity_cancer",
    "auroc",
    "auprc",
    "ece",
    "brier",
    "nll",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "predicted_positive_rate",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    text_frame = frame.copy()
    for column in text_frame.columns:
        text_frame[column] = text_frame[column].map(
            lambda value: f"{value:.4f}" if isinstance(value, float) else str(value)
        )
    header = "| " + " | ".join(text_frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(text_frame.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in text_frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _mean_std(frame: pd.DataFrame, group_columns: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, Any] = dict(zip(group_columns, keys, strict=False))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group.columns else len(group)
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _cohort_statistics(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"variable": "n_images", "level": "all", "n": int(len(metadata))},
        {
            "variable": "label_cancer_binary",
            "level": "positive",
            "n": int((metadata["label_cancer_binary"].astype(int) == 1).sum()),
        },
        {
            "variable": "label_cancer_binary",
            "level": "negative",
            "n": int((metadata["label_cancer_binary"].astype(int) == 0).sum()),
        },
    ]
    for column in ["label_original", "modality", "pid", "split_source"]:
        if column not in metadata.columns:
            continue
        for level, count in metadata[column].value_counts(dropna=False).sort_index().items():
            rows.append({"variable": column, "level": str(level), "n": int(count)})
    return pd.DataFrame(rows)


def _subset_predictions(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    subset_ids = set(metadata["sample_id"].astype(str))
    missing = sorted(subset_ids - set(predictions["sample_id"].astype(str)))
    if missing:
        preview = ", ".join(missing[:5])
        msg = f"Missing {len(missing)} official-test sample_id values in frozen predictions: {preview}"
        raise RuntimeError(msg)
    merged = metadata.merge(predictions, on="sample_id", how="left", suffixes=("_meta", ""))
    if merged[["prob_0", "prob_1"]].isna().any(axis=None):
        msg = "Official-test subset contains missing probabilities after merge."
        raise RuntimeError(msg)
    merged["y_true"] = merged["label_cancer_binary"].astype(int)
    return merged


def _threshold_metrics(
    subset: pd.DataFrame,
    thresholds: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    labels = subset["y_true"].astype(int).to_numpy()
    probabilities = subset[["prob_0", "prob_1"]].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for threshold_name, threshold in thresholds.items():
        scores = probabilities[:, 1]
        y_pred = (scores >= float(threshold)).astype(int)
        metrics = compute_classification_metrics(
            labels,
            y_pred=y_pred,
            y_score=probabilities,
            class_names=["non_cancer", "cancer"],
        )
        confusion = binary_confusion_at_threshold(labels, scores, float(threshold))
        row: dict[str, Any] = {
            "seed": seed,
            "threshold_name": threshold_name,
            "threshold": float(threshold),
        }
        row.update(metrics)
        row.update(confusion)
        row["predicted_positive_rate"] = float(y_pred.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _metrics_by_ebtc_class(
    subset: pd.DataFrame,
    thresholds: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scores = subset["prob_1"].to_numpy(dtype=float)
    working = subset.copy().reset_index(drop=True)
    for threshold_name, threshold in thresholds.items():
        working["threshold_prediction"] = (scores >= float(threshold)).astype(int)
        for label, group in working.groupby("label_original", dropna=False):
            true_positive_class = int(group["label_cancer_binary"].iloc[0]) == 1
            predicted_cancer_rate = float(group["threshold_prediction"].mean())
            rows.append(
                {
                    "seed": seed,
                    "threshold_name": threshold_name,
                    "threshold": float(threshold),
                    "ebtc_class": str(label),
                    "n": int(len(group)),
                    "sensitivity": predicted_cancer_rate if true_positive_class else float("nan"),
                    "specificity": 1.0 - predicted_cancer_rate
                    if not true_positive_class
                    else float("nan"),
                    "false_positive_rate": predicted_cancer_rate
                    if not true_positive_class
                    else float("nan"),
                    "false_negative_rate": 1.0 - predicted_cancer_rate
                    if true_positive_class
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _metrics_by_modality(
    subset: pd.DataFrame,
    thresholds: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold_name, threshold in thresholds.items():
        for modality, group in subset.groupby("modality", dropna=False):
            labels = group["y_true"].astype(int).to_numpy()
            probabilities = group[["prob_0", "prob_1"]].to_numpy(dtype=float)
            y_pred = (probabilities[:, 1] >= float(threshold)).astype(int)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                metrics = compute_classification_metrics(
                    labels,
                    y_pred=y_pred,
                    y_score=probabilities,
                    class_names=["non_cancer", "cancer"],
                )
            confusion = binary_confusion_at_threshold(labels, probabilities[:, 1], float(threshold))
            row: dict[str, Any] = {
                "seed": seed,
                "threshold_name": threshold_name,
                "threshold": float(threshold),
                "modality": str(modality),
                "n": int(len(group)),
                "n_positive": int(labels.sum()),
                "n_negative": int(len(labels) - labels.sum()),
            }
            row.update(metrics)
            row.update(confusion)
            rows.append(row)
    return pd.DataFrame(rows)


def _load_all_image_metrics(final_dir: Path) -> pd.DataFrame:
    path = final_dir / "frozen_validation_metrics_by_seed.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame[frame["dataset"].astype(str) == "ebtc_external"].copy()


def _write_report(
    output: Path,
    cohort: pd.DataFrame,
    summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    all_vs_official: pd.DataFrame,
) -> None:
    default_summary = summary[summary["threshold_name"].astype(str) == "default_0.5"].copy()
    lines = [
        "# EBTC Official Test Split Sensitivity Analysis",
        "",
        "This analysis reused the frozen CDG-Cysto checkpoints, CystoDS-fitted temperatures, and CystoDS-selected thresholds. EBTC was not used for model selection, calibration, or threshold selection.",
        "",
        "## Official Test Split Cohort",
        "",
        _to_markdown_table(cohort),
        "",
        "## Default-Threshold Summary",
        "",
        _to_markdown_table(default_summary),
        "",
        "## All Thresholds Summary",
        "",
        _to_markdown_table(summary),
        "",
        "## EBTC Class-Stratified Summary",
        "",
        _to_markdown_table(class_summary),
        "",
    ]
    if not all_vs_official.empty:
        lines.extend(
            [
                "## All-Image Versus Official-Test EBTC",
                "",
                _to_markdown_table(all_vs_official),
                "",
            ]
        )
    (output / "ebtc_official_test_sensitivity_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-dir",
        default="outputs/v2/final_validation/cdg_cysto_v2_swin_tiny_ce_sampler_modality_aware_v1",
    )
    parser.add_argument(
        "--official-test-split",
        default="outputs/v2/splits/ebtc_external_official_test.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/v2/sensitivity/ebtc_official_test",
    )
    args = parser.parse_args()

    final_dir = Path(args.final_dir)
    metadata = pd.read_csv(args.official_test_split)
    metadata = metadata[metadata["is_valid"].astype(bool)].reset_index(drop=True)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    cohort = _cohort_statistics(metadata)
    write_table(cohort, output / "cohort_statistics.csv")

    metric_frames: list[pd.DataFrame] = []
    class_frames: list[pd.DataFrame] = []
    modality_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for seed_dir in sorted(final_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        seed = int(seed_dir.name.replace("seed_", ""))
        predictions_path = seed_dir / "ebtc_external" / "predictions.csv"
        thresholds_path = seed_dir / "calibration" / "thresholds.json"
        if not predictions_path.exists() or not thresholds_path.exists():
            continue
        predictions = pd.read_csv(predictions_path)
        thresholds = {key: float(value) for key, value in _read_json(thresholds_path).items()}
        subset = _subset_predictions(predictions, metadata)
        subset["seed"] = seed
        prediction_frames.append(subset)
        metric_frames.append(_threshold_metrics(subset, thresholds, seed))
        class_frames.append(_metrics_by_ebtc_class(subset, thresholds, seed))
        modality_frames.append(_metrics_by_modality(subset, thresholds, seed))

    if not metric_frames:
        msg = f"No frozen seed predictions found under {final_dir}"
        raise RuntimeError(msg)

    metrics = pd.concat(metric_frames, ignore_index=True)
    class_metrics = pd.concat(class_frames, ignore_index=True)
    modality_metrics = pd.concat(modality_frames, ignore_index=True)
    predictions_subset = pd.concat(prediction_frames, ignore_index=True)

    write_table(predictions_subset, output / "predictions_official_test_by_seed.csv")
    write_table(metrics, output / "metrics_by_seed.csv")
    write_table(class_metrics, output / "metrics_by_ebtc_class_by_seed.csv")
    write_table(modality_metrics, output / "metrics_by_modality_by_seed.csv")

    summary = _mean_std(metrics, ["threshold_name"], ["threshold", *METRIC_COLUMNS])
    class_summary = _mean_std(
        class_metrics,
        ["threshold_name", "ebtc_class"],
        ["n", "sensitivity", "specificity", "false_positive_rate", "false_negative_rate"],
    )
    modality_summary = _mean_std(
        modality_metrics,
        ["threshold_name", "modality"],
        ["n", "n_positive", "n_negative", *METRIC_COLUMNS],
    )
    write_table(summary, output / "metrics_summary.csv")
    write_table(class_summary, output / "metrics_by_ebtc_class_summary.csv")
    write_table(modality_summary, output / "metrics_by_modality_summary.csv")

    all_image = _load_all_image_metrics(final_dir)
    all_vs_official = pd.DataFrame()
    if not all_image.empty:
        comparison_rows: list[pd.DataFrame] = []
        all_default = all_image[all_image["threshold_name"].astype(str) == "default_0.5"].copy()
        official_default = metrics[metrics["threshold_name"].astype(str) == "default_0.5"].copy()
        all_default["ebtc_cohort"] = "all_image_harmonized"
        official_default["ebtc_cohort"] = "official_test_split"
        common_columns = [
            "ebtc_cohort",
            "seed",
            "threshold_name",
            "balanced_accuracy",
            "macro_f1",
            "sensitivity_cancer",
            "specificity_cancer",
            "auroc",
            "auprc",
        ]
        comparison_rows.extend(
            [all_default[common_columns], official_default[common_columns]]
        )
        all_vs_official_seed = pd.concat(comparison_rows, ignore_index=True)
        write_table(all_vs_official_seed, output / "all_image_vs_official_test_by_seed.csv")
        all_vs_official = _mean_std(
            all_vs_official_seed,
            ["ebtc_cohort", "threshold_name"],
            [
                "balanced_accuracy",
                "macro_f1",
                "sensitivity_cancer",
                "specificity_cancer",
                "auroc",
                "auprc",
            ],
        )
        write_table(all_vs_official, output / "all_image_vs_official_test_summary.csv")

    _write_report(output, cohort, summary, class_summary, all_vs_official)
    print(f"Wrote EBTC official-test sensitivity analysis to {output}")


if __name__ == "__main__":
    main()
