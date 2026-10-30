"""Finalize CDG-Cysto v2 frozen analyses and paper-ready tables."""

# ruff: noqa: I001, E402

from __future__ import annotations

import argparse
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cdg_cysto.eval.calibration import TemperatureScaler, apply_temperature_np, calibration_metrics
from cdg_cysto.eval.metrics import (
    compute_classification_metrics,
    ensure_probabilities,
    save_confusion_matrix_plot,
    save_reliability_diagram,
)
from cdg_cysto.eval.thresholding import binary_confusion_at_threshold, select_standard_thresholds
from cdg_cysto.utils.io import write_table
from scripts.v2_external_validate import _collect


LABEL_COLUMN = "label_cancer_binary"
CLASS_NAMES = ["non_cancer", "cancer"]
FINAL_METHOD_DIR = Path(
    "outputs/v2/final_validation/cdg_cysto_v2_swin_tiny_ce_sampler_modality_aware_v1"
)
BASELINE_OUT = Path("outputs/v2/final_validation/frozen_baselines")
ERROR_OUT = Path("outputs/v2/error_analysis/final_ebtc")
MODALITY_OUT = Path("outputs/v2/modality_analysis")
BOOTSTRAP_OUT = Path("outputs/v2/bootstrap")
TABLE_OUT = Path("outputs/v2/tables")
REPORT_OUT = Path("outputs/v2/reports/experiment_report_v2_final.md")


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    leaderboard: str
    model_name: str
    variant_id: str
    augmentation_name: str
    seeds: tuple[int, ...]


METHODS = [
    MethodSpec(
        "B0",
        "EfficientNet-B3 + CE",
        "stage04",
        "efficientnet_b3",
        "ce_none",
        "basic",
        (42,),
    ),
    MethodSpec(
        "B1",
        "ConvNeXt-Tiny + CE",
        "stage05",
        "convnext_tiny",
        "ce_none",
        "basic",
        (42, 2025, 3407),
    ),
    MethodSpec("B2", "Swin-Tiny + CE", "stage04", "swin_tiny", "ce_none", "basic", (42,)),
    MethodSpec(
        "B3",
        "CDG-Cysto v2",
        "stage06",
        "swin_tiny",
        "ce_class_balanced_sampler",
        "modality_aware_v1",
        (42, 2025, 3407),
    ),
    MethodSpec(
        "B4",
        "ConvNeXt-Tiny + Logit Adjustment tau=0.5",
        "stage04",
        "convnext_tiny",
        "logit_adjustment_tau_0_5",
        "basic",
        (42,),
    ),
    MethodSpec(
        "B5",
        "Swin-Tiny + Balanced Softmax",
        "stage05",
        "swin_tiny",
        "balanced_softmax_none",
        "basic",
        (42, 2025, 3407),
    ),
]


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _method_dir(spec: MethodSpec) -> Path:
    return BASELINE_OUT / f"{spec.method_id}_{_safe_slug(spec.display_name)}"


def _filter_formal(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.copy()
    if "run_type" in filtered.columns:
        filtered = filtered[filtered["run_type"].astype(str).eq("formal")].copy()
    for column in ("run_id", "run_dir"):
        if column in filtered.columns:
            filtered = filtered[
                ~filtered[column].astype(str).str.contains("smoke", case=False, na=False)
            ].copy()
    return filtered


def _valid_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "is_valid" in frame.columns:
        valid = frame["is_valid"].astype(str).str.lower().isin({"true", "1", "yes"})
        frame = frame[valid].copy()
    return frame.reset_index(drop=True)


def _logit_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith("logit_")],
        key=lambda value: int(value.split("_")[1]),
    )


def _write_logits(sample_ids: list[str], labels: np.ndarray, logits: np.ndarray, path: Path) -> None:
    frame = pd.DataFrame({"sample_id": sample_ids, "y_true": labels})
    for index in range(logits.shape[1]):
        frame[f"logit_{index}"] = logits[:, index]
    write_table(frame, path)


def _read_logits(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    columns = _logit_columns(frame)
    return (
        frame["sample_id"].astype(str).tolist(),
        frame["y_true"].to_numpy(dtype=int),
        frame[columns].to_numpy(dtype=float),
    )


def _prediction_frame(
    sample_ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    y_pred = (probabilities[:, 1] >= threshold).astype(int)
    frame = pd.DataFrame({"sample_id": sample_ids, "y_true": labels, "y_pred": y_pred})
    for index in range(probabilities.shape[1]):
        frame[f"prob_{index}"] = probabilities[:, index]
    frame["confidence"] = np.maximum(probabilities[:, 0], probabilities[:, 1])
    frame["cancer_score"] = probabilities[:, 1]
    return frame


def _load_leaderboards() -> dict[str, pd.DataFrame]:
    return {
        "stage04": _filter_formal(pd.read_csv("outputs/v2/leaderboards/stage04_imbalance_leaderboard.csv")),
        "stage05": _filter_formal(pd.read_csv("outputs/v2/leaderboards/stage05_multiseed_leaderboard.csv")),
        "stage06": _filter_formal(pd.read_csv("outputs/v2/leaderboards/stage06_augmentation_leaderboard.csv")),
    }


def _run_row(spec: MethodSpec, seed: int, leaderboards: dict[str, pd.DataFrame]) -> pd.Series:
    frame = leaderboards[spec.leaderboard].copy()
    frame["seed"] = frame["seed"].astype(int)
    selected = frame[
        (frame["model_name"].astype(str) == spec.model_name)
        & (frame["variant_id"].astype(str) == spec.variant_id)
        & (frame["augmentation_name"].astype(str) == spec.augmentation_name)
        & (frame["seed"] == seed)
    ]
    if selected.empty:
        msg = f"Missing checkpoint row for {spec.method_id} seed {seed}"
        raise RuntimeError(msg)
    return selected.iloc[0]


def _checkpoint(row: pd.Series) -> Path:
    checkpoint = Path(str(row["run_dir"])) / "checkpoint_best.pt"
    if not checkpoint.exists():
        msg = f"Missing checkpoint: {checkpoint}"
        raise FileNotFoundError(msg)
    return checkpoint


def _collect_logits_cached(
    checkpoint: Path,
    split_name: str,
    split_path: Path,
    seed_dir: Path,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    target = seed_dir / split_name / "logits_raw.csv"
    if target.exists():
        return _read_logits(target)
    metadata = _valid_metadata(split_path)
    labels = metadata[LABEL_COLUMN].astype(int).to_numpy()
    logits, sample_ids = _collect(checkpoint, metadata, LABEL_COLUMN)
    _write_logits(sample_ids, labels, logits, target)
    write_table(_prediction_frame(sample_ids, labels, ensure_probabilities(logits), 0.5), seed_dir / split_name / "predictions_raw_default_0.5.csv")
    return sample_ids, labels, logits


def _threshold_rows(
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for name, threshold in thresholds.items():
        row = {"threshold_name": name}
        row.update(binary_confusion_at_threshold(labels, scores, threshold))
        rows.append(row)
    return rows


def _metric_row(
    spec: MethodSpec,
    seed: int,
    dataset: str,
    threshold_name: str,
    threshold: float,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    y_pred = (probabilities[:, 1] >= threshold).astype(int)
    metrics = compute_classification_metrics(
        labels,
        y_pred=y_pred,
        y_score=probabilities,
        class_names=CLASS_NAMES,
    )
    row: dict[str, Any] = {
        "method_id": spec.method_id,
        "method": spec.display_name,
        "seed": seed,
        "dataset": dataset,
        "threshold_name": threshold_name,
        "threshold": threshold,
    }
    row.update(metrics)
    row.update(binary_confusion_at_threshold(labels, probabilities[:, 1], threshold))
    return row


def _calibration_row(
    spec: MethodSpec,
    seed: int,
    dataset: str,
    temperature: float,
    labels: np.ndarray,
    raw_logits: np.ndarray,
) -> dict[str, Any]:
    before = calibration_metrics(labels, raw_logits)
    after = calibration_metrics(labels, apply_temperature_np(raw_logits, temperature))
    row: dict[str, Any] = {
        "method_id": spec.method_id,
        "method": spec.display_name,
        "seed": seed,
        "dataset": dataset,
        "temperature": temperature,
    }
    for metric in ("ece", "brier", "nll"):
        row[f"before_{metric}"] = before[metric]
        row[f"after_{metric}"] = after[metric]
        row[f"delta_after_minus_before_{metric}"] = after[metric] - before[metric]
        row[f"improved_{metric}"] = bool(after[metric] < before[metric])
    return row


def _mean_std(frame: pd.DataFrame, group_columns: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=False))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group.columns else len(group)
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run_frozen_baselines(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = {
        "cystods_calib": Path(args.calib_split),
        "cystods_internal": Path(args.internal_split),
        "ebtc_external": Path(args.external_split),
    }
    leaderboards = _load_leaderboards()
    selected_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    threshold_metric_rows: list[dict[str, Any]] = []

    for spec in METHODS:
        for seed in spec.seeds:
            row = _run_row(spec, seed, leaderboards)
            selected_rows.append({"method_id": spec.method_id, "method": spec.display_name, **row.to_dict()})
            seed_dir = _method_dir(spec) / f"seed_{seed}"
            checkpoint = _checkpoint(row)

            sample_ids, labels, calib_logits = _collect_logits_cached(
                checkpoint,
                "cystods_calib",
                splits["cystods_calib"],
                seed_dir,
            )
            scaler = TemperatureScaler().fit(calib_logits, labels)
            calib_probabilities = ensure_probabilities(scaler.transform(calib_logits))
            thresholds = {
                key: value
                for key, value in select_standard_thresholds(labels, calib_probabilities[:, 1]).items()
                if key in {"default_0.5", "youden", "sensitivity_90"}
            }
            write_table(pd.DataFrame(_threshold_rows(labels, calib_probabilities[:, 1], thresholds)), seed_dir / "threshold_metrics.csv")
            (seed_dir / "temperature.txt").write_text(f"{scaler.temperature}\n", encoding="utf-8")

            for split_name, split_path in splits.items():
                split_sample_ids, split_labels, split_logits = _collect_logits_cached(
                    checkpoint,
                    split_name,
                    split_path,
                    seed_dir,
                )
                calibration_rows.append(
                    _calibration_row(spec, seed, split_name, scaler.temperature, split_labels, split_logits)
                )
                calibrated_logits = scaler.transform(split_logits)
                _write_logits(
                    split_sample_ids,
                    split_labels,
                    calibrated_logits,
                    seed_dir / split_name / "logits_calibrated.csv",
                )
                probabilities = ensure_probabilities(calibrated_logits)
                for threshold_name, threshold in thresholds.items():
                    predictions = _prediction_frame(split_sample_ids, split_labels, probabilities, threshold)
                    write_table(
                        predictions,
                        seed_dir / split_name / f"predictions_{threshold_name}.csv",
                    )
                    threshold_metric_rows.append(
                        {
                            "method_id": spec.method_id,
                            "method": spec.display_name,
                            "seed": seed,
                            "dataset": split_name,
                            "threshold_name": threshold_name,
                            **binary_confusion_at_threshold(split_labels, probabilities[:, 1], threshold),
                        }
                    )
                    if split_name != "cystods_calib":
                        metric_rows.append(
                            _metric_row(
                                spec,
                                seed,
                                split_name,
                                threshold_name,
                                threshold,
                                split_labels,
                                probabilities,
                            )
                        )

    selected = pd.DataFrame(selected_rows)
    metrics_by_seed = pd.DataFrame(metric_rows)
    calibration_by_seed = pd.DataFrame(calibration_rows)
    threshold_metrics = pd.DataFrame(threshold_metric_rows)
    metric_columns = [
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
        "threshold",
    ]
    comparison = _mean_std(
        metrics_by_seed,
        ["method_id", "method", "dataset", "threshold_name"],
        metric_columns,
    )
    calibration_summary = _mean_std(
        calibration_by_seed,
        ["method_id", "method", "dataset"],
        [
            "temperature",
            "before_ece",
            "after_ece",
            "before_brier",
            "after_brier",
            "before_nll",
            "after_nll",
        ],
    )
    write_table(selected, BASELINE_OUT / "selected_frozen_runs.csv")
    write_table(metrics_by_seed, BASELINE_OUT / "frozen_baseline_comparison_by_seed.csv")
    write_table(comparison, BASELINE_OUT / "frozen_baseline_comparison.csv")
    write_table(comparison, Path("outputs/v2/metrics/frozen_baseline_comparison.csv"))
    write_table(calibration_by_seed, BASELINE_OUT / "calibration_before_after_by_seed.csv")
    write_table(calibration_summary, BASELINE_OUT / "calibration_before_after.csv")
    write_table(threshold_metrics, BASELINE_OUT / "threshold_metrics_by_seed.csv")
    return comparison, calibration_summary


def _load_thresholds_for_seed(method_id: str, seed: int) -> dict[str, float]:
    frame = pd.read_csv(BASELINE_OUT / f"{method_id}_{_safe_slug(_method_name(method_id))}" / f"seed_{seed}" / "threshold_metrics.csv")
    return dict(zip(frame["threshold_name"].astype(str), frame["threshold"].astype(float), strict=False))


def _method_name(method_id: str) -> str:
    for spec in METHODS:
        if spec.method_id == method_id:
            return spec.display_name
    return method_id


def _resolve_image_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run_ebtc_error_analysis() -> pd.DataFrame:
    metadata = _valid_metadata(Path("outputs/v2/splits/ebtc_external_all.csv"))
    spec = next(item for item in METHODS if item.method_id == "B3")
    merged_rows: list[pd.DataFrame] = []
    for seed in spec.seeds:
        seed_dir = _method_dir(spec) / f"seed_{seed}"
        thresholds = pd.read_csv(seed_dir / "threshold_metrics.csv")
        for _, threshold_row in thresholds.iterrows():
            threshold_name = str(threshold_row["threshold_name"])
            predictions = pd.read_csv(seed_dir / "ebtc_external" / f"predictions_{threshold_name}.csv")
            merged = predictions.merge(metadata, on="sample_id", how="left", suffixes=("", "_meta"))
            merged["seed"] = seed
            merged["threshold_name"] = threshold_name
            merged["is_correct"] = merged["y_true"].astype(int) == merged["y_pred"].astype(int)
            merged["is_false_negative"] = (merged["y_true"].astype(int) == 1) & (
                merged["y_pred"].astype(int) == 0
            )
            merged["is_false_positive"] = (merged["y_true"].astype(int) == 0) & (
                merged["y_pred"].astype(int) == 1
            )
            merged_rows.append(merged)
    merged_all = pd.concat(merged_rows, ignore_index=True)

    def grouped(group_columns: list[str]) -> pd.DataFrame:
        rows = []
        for keys, group in merged_all.groupby(group_columns, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_columns, keys, strict=False))
            row["n_predictions"] = len(group)
            row["n_unique_samples"] = group["sample_id"].nunique()
            row["errors"] = int((~group["is_correct"]).sum())
            row["error_rate"] = row["errors"] / len(group) if len(group) else math.nan
            row["false_negatives"] = int(group["is_false_negative"].sum())
            row["false_positives"] = int(group["is_false_positive"].sum())
            rows.append(row)
        return pd.DataFrame(rows)

    high_confidence = merged_all[~merged_all["is_correct"]].copy()
    high_confidence = high_confidence.sort_values("confidence", ascending=False)
    cases = {
        "hgc_false_negatives": merged_all[
            merged_all["is_false_negative"] & merged_all["label_original"].astype(str).eq("HGC")
        ],
        "lgc_false_negatives": merged_all[
            merged_all["is_false_negative"] & merged_all["label_original"].astype(str).eq("LGC")
        ],
        "ntl_false_positives": merged_all[
            merged_all["is_false_positive"] & merged_all["label_original"].astype(str).eq("NTL")
        ],
        "nst_false_positives": merged_all[
            merged_all["is_false_positive"] & merged_all["label_original"].astype(str).eq("NST")
        ],
        "high_confidence_errors": high_confidence,
    }
    ERROR_OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in cases.items():
        write_table(frame, ERROR_OUT / f"{name}.csv")
    error_by_class = grouped(["threshold_name", "label_original"])
    error_by_modality = grouped(["threshold_name", "modality"])
    write_table(error_by_class, ERROR_OUT / "error_by_ebtc_class.csv")
    write_table(error_by_modality, ERROR_OUT / "error_by_modality.csv")

    image_dir = ERROR_OUT / "error_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for rank, (_, row) in enumerate(high_confidence.head(60).iterrows(), start=1):
        source = _resolve_image_path(str(row["image_path"]))
        if not source.exists():
            continue
        target = (
            image_dir
            / f"{rank:03d}_seed{row['seed']}_{row['threshold_name']}_{row['label_original']}_{row['sample_id']}{source.suffix.lower()}"
        )
        shutil.copy2(source, target)
    return error_by_class


def _threshold_sweep(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.0, 1.0, 101):
        y_pred = (scores >= threshold).astype(int)
        confusion = binary_confusion_at_threshold(labels, scores, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": confusion["sensitivity"],
                "specificity": confusion["specificity"],
                "balanced_accuracy": (confusion["sensitivity"] + confusion["specificity"]) / 2,
                "macro_f1": float(f1_score(labels, y_pred, average="macro", zero_division=0)),
                "youden_index": confusion["sensitivity"] + confusion["specificity"] - 1.0,
            }
        )
    return pd.DataFrame(rows)


def run_modality_analysis() -> pd.DataFrame:
    metadata = _valid_metadata(Path("outputs/v2/splits/cystods_test_internal.csv"))
    spec = next(item for item in METHODS if item.method_id == "B3")
    sample_counts = (
        metadata.groupby(["modality", LABEL_COLUMN], dropna=False)
        .size()
        .reset_index(name="n")
        .rename(columns={LABEL_COLUMN: "label_cancer_binary"})
    )
    MODALITY_OUT.mkdir(parents=True, exist_ok=True)
    write_table(sample_counts, MODALITY_OUT / "blc_wlc_sample_counts.csv")
    confusion_rows = []
    sweep_rows = []
    reliability_dir = MODALITY_OUT / "reliability_diagrams"
    confusion_dir = MODALITY_OUT / "confusion_matrices"
    for seed in spec.seeds:
        seed_dir = _method_dir(spec) / f"seed_{seed}"
        thresholds = pd.read_csv(seed_dir / "threshold_metrics.csv")
        logits_frame = pd.read_csv(seed_dir / "cystods_internal" / "logits_calibrated.csv")
        probabilities = ensure_probabilities(logits_frame[_logit_columns(logits_frame)].to_numpy(dtype=float))
        predictions_base = pd.DataFrame(
            {
                "sample_id": logits_frame["sample_id"].astype(str),
                "y_true": logits_frame["y_true"].astype(int),
                "prob_0": probabilities[:, 0],
                "prob_1": probabilities[:, 1],
            }
        ).merge(metadata[["sample_id", "modality"]], on="sample_id", how="left")
        for modality, group in predictions_base.groupby("modality", dropna=False):
            labels = group["y_true"].to_numpy(dtype=int)
            scores = group["prob_1"].to_numpy(dtype=float)
            probs = group[["prob_0", "prob_1"]].to_numpy(dtype=float)
            sweep = _threshold_sweep(labels, scores)
            sweep["seed"] = seed
            sweep["modality"] = modality
            sweep_rows.append(sweep)
            if len(np.unique(labels)) >= 1:
                save_reliability_diagram(
                    labels,
                    probs,
                    reliability_dir / f"seed{seed}_{modality}_reliability.png",
                )
            for _, threshold_row in thresholds.iterrows():
                threshold_name = str(threshold_row["threshold_name"])
                threshold = float(threshold_row["threshold"])
                y_pred = (scores >= threshold).astype(int)
                metrics = compute_classification_metrics(
                    labels,
                    y_pred=y_pred,
                    y_score=probs,
                    class_names=CLASS_NAMES,
                )
                confusion = binary_confusion_at_threshold(labels, scores, threshold)
                row: dict[str, Any] = {
                    "seed": seed,
                    "modality": modality,
                    "threshold_name": threshold_name,
                    "threshold": threshold,
                    "n": len(group),
                }
                row.update(metrics)
                row.update(confusion)
                row["blc_specificity_warning"] = bool(
                    str(modality).upper() == "BLC"
                    and threshold_name == "default_0.5"
                    and confusion["specificity"] < 0.5
                )
                confusion_rows.append(row)
                if threshold_name in {"default_0.5", "sensitivity_90"}:
                    save_confusion_matrix_plot(
                        labels,
                        y_pred,
                        CLASS_NAMES,
                        confusion_dir / f"seed{seed}_{modality}_{threshold_name}.png",
                    )
    confusion_frame = pd.DataFrame(confusion_rows)
    sweep_frame = pd.concat(sweep_rows, ignore_index=True)
    summary = _mean_std(
        confusion_frame,
        ["modality", "threshold_name"],
        ["sensitivity_cancer", "specificity_cancer", "balanced_accuracy", "macro_f1", "auroc", "auprc"],
    )
    write_table(confusion_frame, MODALITY_OUT / "blc_wlc_confusion_matrix.csv")
    write_table(sweep_frame, MODALITY_OUT / "blc_wlc_threshold_sweep.csv")
    write_table(summary, MODALITY_OUT / "blc_wlc_modality_summary.csv")
    blc_default = summary[
        summary["modality"].astype(str).str.upper().eq("BLC")
        & summary["threshold_name"].astype(str).eq("default_0.5")
    ]
    warning = "BLC default-threshold specificity is low."
    if not blc_default.empty:
        warning = (
            f"BLC default-threshold specificity is low: "
            f"{float(blc_default['specificity_cancer_mean'].iloc[0]):.3f}."
        )
    (MODALITY_OUT / "modality_analysis.md").write_text(
        "# BLC/WLC Modality Analysis\n\n"
        f"{warning}\n\n"
        "See `blc_wlc_confusion_matrix.csv`, `blc_wlc_threshold_sweep.csv`, "
        "and `reliability_diagrams/`.\n",
        encoding="utf-8",
    )
    return summary


def _bootstrap_metric_values(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    metric: str,
    *,
    n_bootstraps: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(labels), size=len(labels))
        y_true = labels[indices]
        y_score = scores[indices]
        y_pred = predictions[indices]
        try:
            if metric == "auroc":
                if len(np.unique(y_true)) < 2:
                    continue
                value = roc_auc_score(y_true, y_score)
            elif metric == "auprc":
                if len(np.unique(y_true)) < 2:
                    continue
                value = average_precision_score(y_true, y_score)
            elif metric == "macro_f1":
                value = f1_score(y_true, y_pred, average="macro", zero_division=0)
            elif metric == "sensitivity":
                conf = binary_confusion_at_threshold(y_true, y_pred.astype(float), 0.5)
                value = conf["sensitivity"]
            elif metric == "specificity":
                conf = binary_confusion_at_threshold(y_true, y_pred.astype(float), 0.5)
                value = conf["specificity"]
            else:
                continue
        except ValueError:
            continue
        if not math.isnan(float(value)):
            values.append(float(value))
    if not values:
        return {"mean": math.nan, "lower": math.nan, "upper": math.nan}
    return {
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
    }


def run_bootstrap_ci(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    comparison = pd.read_csv(BASELINE_OUT / "frozen_baseline_comparison_by_seed.csv")
    for _, row in comparison.iterrows():
        if row["dataset"] not in {"cystods_internal", "ebtc_external"}:
            continue
        spec = next(item for item in METHODS if item.method_id == row["method_id"])
        seed = int(row["seed"])
        threshold_name = str(row["threshold_name"])
        pred_path = _method_dir(spec) / f"seed_{seed}" / str(row["dataset"]) / f"predictions_{threshold_name}.csv"
        predictions = pd.read_csv(pred_path)
        labels = predictions["y_true"].to_numpy(dtype=int)
        scores = predictions["prob_1"].to_numpy(dtype=float)
        y_pred = predictions["y_pred"].to_numpy(dtype=int)
        for metric in ("auroc", "auprc", "macro_f1", "sensitivity", "specificity"):
            ci = _bootstrap_metric_values(
                labels,
                scores,
                y_pred,
                metric,
                n_bootstraps=args.n_bootstraps,
                seed=42 + seed,
            )
            rows.append(
                {
                    "method_id": spec.method_id,
                    "method": spec.display_name,
                    "seed": seed,
                    "dataset": row["dataset"],
                    "threshold_name": threshold_name,
                    "metric": metric,
                    **ci,
                }
            )
    frame = pd.DataFrame(rows)
    write_table(frame, BOOTSTRAP_OUT / "bootstrap_ci.csv")
    write_table(frame, Path("outputs/v2/metrics/bootstrap_ci.csv"))
    return frame


def write_paper_tables() -> None:
    TABLE_OUT.mkdir(parents=True, exist_ok=True)
    split_rows = []
    split_paths = {
        "CystoDS train": "outputs/v2/splits/cystods_train.csv",
        "CystoDS val": "outputs/v2/splits/cystods_val.csv",
        "CystoDS calib": "outputs/v2/splits/cystods_calib.csv",
        "CystoDS internal test": "outputs/v2/splits/cystods_test_internal.csv",
        "EBTC external": "outputs/v2/splits/ebtc_external_all.csv",
    }
    for name, path in split_paths.items():
        frame = _valid_metadata(Path(path))
        row: dict[str, Any] = {
            "split": name,
            "n": len(frame),
            "positive": int((frame[LABEL_COLUMN].astype(int) == 1).sum()),
            "negative": int((frame[LABEL_COLUMN].astype(int) == 0).sum()),
            "patients": frame["pid"].nunique() if "pid" in frame.columns else math.nan,
        }
        if "modality" in frame.columns:
            row["modalities"] = "; ".join(
                f"{key}:{value}" for key, value in frame["modality"].value_counts().sort_index().items()
            )
        if "label_original" in frame.columns:
            row["labels"] = "; ".join(
                f"{key}:{value}"
                for key, value in frame["label_original"].value_counts().sort_index().items()
            )
        split_rows.append(row)
    write_table(pd.DataFrame(split_rows), TABLE_OUT / "table1_dataset_split_statistics.csv")

    stage04 = _filter_formal(pd.read_csv("outputs/v2/leaderboards/stage04_imbalance_leaderboard.csv"))
    columns = [
        "model_name",
        "variant_id",
        "loss_name",
        "sampler_name",
        "val_auroc",
        "val_auprc",
        "val_macro_f1",
        "val_sensitivity_cancer",
        "val_specificity_cancer",
        "val_ece",
    ]
    write_table(stage04[[column for column in columns if column in stage04.columns]], TABLE_OUT / "table2_stage04_validation_comparison.csv")
    stage06 = pd.read_csv("outputs/v2/leaderboards/stage06_augmentation_summary.csv")
    write_table(stage06, TABLE_OUT / "table3_stage06_augmentation_comparison.csv")

    baseline = pd.read_csv(BASELINE_OUT / "frozen_baseline_comparison.csv")
    write_table(
        baseline[baseline["dataset"].eq("cystods_internal")],
        TABLE_OUT / "table4_frozen_baseline_cystods_internal.csv",
    )
    write_table(
        baseline[baseline["dataset"].eq("ebtc_external")],
        TABLE_OUT / "table5_frozen_baseline_ebtc_external.csv",
    )
    thresholds = pd.read_csv(BASELINE_OUT / "threshold_metrics_by_seed.csv")
    write_table(thresholds[thresholds["method_id"].eq("B3")], TABLE_OUT / "table6_threshold_analysis.csv")
    write_table(pd.read_csv(ERROR_OUT / "error_by_ebtc_class.csv"), TABLE_OUT / "table7_ebtc_class_error_analysis.csv")
    write_table(pd.read_csv(MODALITY_OUT / "blc_wlc_modality_summary.csv"), TABLE_OUT / "table8_modality_stratified_analysis.csv")
    write_table(pd.read_csv(BASELINE_OUT / "calibration_before_after.csv"), TABLE_OUT / "table9_calibration_before_after.csv")


def _to_markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: f"{value:.4f}" if isinstance(value, float) else str(value)
        )
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_final_report() -> None:
    stage06_text = Path("outputs/v2/reports/stage06_augmentation_analysis.md").read_text(encoding="utf-8")
    final_text = (FINAL_METHOD_DIR / "final_validation_cdg_cysto_v2.md").read_text(encoding="utf-8")
    comparison = pd.read_csv(BASELINE_OUT / "frozen_baseline_comparison.csv")
    calibration = pd.read_csv(BASELINE_OUT / "calibration_before_after.csv")
    modality = pd.read_csv(MODALITY_OUT / "blc_wlc_modality_summary.csv")
    status_rows = [
        {"stage": "Stage 04 imbalance comparison", "status": "PASS", "artifact": "outputs/v2/leaderboards/stage04_imbalance_leaderboard.csv"},
        {"stage": "Stage 06 augmentation comparison", "status": "PASS", "artifact": "outputs/v2/reports/stage06_augmentation_analysis.md"},
        {"stage": "CystoDS internal frozen validation", "status": "PASS", "artifact": str(FINAL_METHOD_DIR / "frozen_validation_metrics_summary.csv")},
        {"stage": "EBTC external frozen validation", "status": "PASS", "artifact": str(FINAL_METHOD_DIR / "frozen_validation_metrics_summary.csv")},
        {"stage": "Frozen baseline comparison", "status": "PASS", "artifact": str(BASELINE_OUT / "frozen_baseline_comparison.csv")},
        {"stage": "Calibration before/after", "status": "WARNING", "artifact": str(BASELINE_OUT / "calibration_before_after.csv")},
        {"stage": "EBTC error analysis", "status": "PASS", "artifact": str(ERROR_OUT / "high_confidence_errors.csv")},
        {"stage": "BLC/WLC modality analysis", "status": "WARNING", "artifact": str(MODALITY_OUT / "modality_analysis.md")},
        {"stage": "Bootstrap confidence intervals", "status": "PASS", "artifact": str(BOOTSTRAP_OUT / "bootstrap_ci.csv")},
        {"stage": "Paper tables", "status": "PASS", "artifact": str(TABLE_OUT)},
    ]
    lines = [
        "# CDG-Cysto v2 Final Experiment Report",
        "",
        "Primary method is frozen as `swin_tiny + ce_class_balanced_sampler + modality_aware_v1`, seeds `[42, 2025, 3407]`.",
        "EBTC external results were not used to modify model structure, loss, augmentation, calibration, or thresholds.",
        "",
        "## Stage Status",
        "",
        _to_markdown_table(pd.DataFrame(status_rows), max_rows=50),
        "",
        "## Stage 06 Source Summary",
        "",
        stage06_text[:5000],
        "",
        "## Final Validation Source Summary",
        "",
        final_text[:5000],
        "",
        "## Frozen Baseline Comparison",
        "",
        _to_markdown_table(
            comparison[
                comparison["threshold_name"].isin(["default_0.5", "sensitivity_90"])
                & comparison["dataset"].isin(["cystods_internal", "ebtc_external"])
            ][
                [
                    "method_id",
                    "method",
                    "dataset",
                    "threshold_name",
                    "balanced_accuracy_mean",
                    "macro_f1_mean",
                    "sensitivity_cancer_mean",
                    "specificity_cancer_mean",
                    "auroc_mean",
                    "auprc_mean",
                ]
            ],
            max_rows=50,
        ),
        "",
        "## Calibration Note",
        "",
        "Temperature scaling results are mixed across datasets and methods; this report does not claim calibration universally improves performance.",
        "",
        _to_markdown_table(
            calibration[
                [
                    "method_id",
                    "method",
                    "dataset",
                    "before_ece_mean",
                    "after_ece_mean",
                    "before_brier_mean",
                    "after_brier_mean",
                    "before_nll_mean",
                    "after_nll_mean",
                ]
            ],
            max_rows=50,
        ),
        "",
        "## BLC/WLC Note",
        "",
        "BLC specificity is low under the default threshold and should be explicitly discussed as a threshold/domain trade-off.",
        "",
        _to_markdown_table(modality, max_rows=50),
        "",
        "## Paper Tables",
        "",
        "\n".join(
            f"- Table {index}: `{path}`"
            for index, path in enumerate(sorted(TABLE_OUT.glob("table*.csv")), start=1)
        ),
        "",
    ]
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-split", default="outputs/v2/splits/cystods_calib.csv")
    parser.add_argument("--internal-split", default="outputs/v2/splits/cystods_test_internal.csv")
    parser.add_argument("--external-split", default="outputs/v2/splits/ebtc_external_all.csv")
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    args = parser.parse_args()

    run_frozen_baselines(args)
    run_ebtc_error_analysis()
    run_modality_analysis()
    run_bootstrap_ci(args)
    write_paper_tables()
    write_final_report()
    print(f"Wrote final report to {REPORT_OUT}")


if __name__ == "__main__":
    main()
