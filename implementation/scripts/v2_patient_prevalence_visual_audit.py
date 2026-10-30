"""Patient-level, prevalence-aware, and visual-audit analyses for CDG-Cysto v2."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdg_cysto.explain.gradcam_vit import overlay_cam_on_image, save_cam_case_figure  # noqa: E402
from cdg_cysto.models.factory import create_model  # noqa: E402
from cdg_cysto.train.engine_v2 import V2MetadataDataset, _build_transform, collate_v2  # noqa: E402
from cdg_cysto.utils.checkpoint import load_checkpoint  # noqa: E402
from cdg_cysto.utils.device import get_device  # noqa: E402
from cdg_cysto.utils.io import write_json, write_table  # noqa: E402

SEEDS = (42, 2025, 3407)
THRESHOLDS = ("default_0.5", "sensitivity_90", "youden")
PATIENT_AGGREGATIONS = (
    "max_probability",
    "top3_mean",
    "top5_mean",
    "mean_probability",
    "positive_image_proportion",
)
METRICS = (
    "AUROC",
    "AUPRC",
    "balanced_accuracy",
    "macro_F1",
    "sensitivity",
    "specificity",
    "PPV",
    "NPV",
)
ASSUMED_PREVALENCES = (0.05, 0.10, 0.20, 0.50)
METHOD_DIR = Path("outputs/v2/final_validation/frozen_baselines/B3_CDG_Cysto_v2")
SPLITS = {
    "cystods_calib": Path("outputs/v2/splits/cystods_calib.csv"),
    "cystods_internal": Path("outputs/v2/splits/cystods_test_internal.csv"),
    "ebtc_external": Path("outputs/v2/splits/ebtc_external_all.csv"),
}
PATIENT_OUT = Path("outputs/v2/patient_level")
PREVALENCE_OUT = Path("outputs/v2/prevalence_analysis")
VISUAL_OUT = Path("outputs/v2/visual_audit")
SOURCE_OUT = Path("outputs/v2/source_tables")
DIAGNOSTIC_OUT = Path("outputs/v2/diagnostics")


@dataclass(frozen=True)
class ThresholdSet:
    default_0_5: float
    sensitivity_90: float
    youden: float

    def as_dict(self) -> dict[str, float]:
        return {
            "default_0.5": self.default_0_5,
            "sensitivity_90": self.sensitivity_90,
            "youden": self.youden,
        }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _logit_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith("logit_")],
        key=lambda value: int(value.split("_")[1]),
    )


def _valid_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "is_valid" in frame.columns:
        valid = frame["is_valid"].astype(str).str.lower().isin({"true", "1", "yes"})
        frame = frame[valid].copy()
    frame["pid"] = frame["pid"].fillna("unknown").astype(str)
    return frame.reset_index(drop=True)


def _read_thresholds(seed: int) -> ThresholdSet:
    frame = pd.read_csv(METHOD_DIR / f"seed_{seed}" / "threshold_metrics.csv")
    values = {
        str(row["threshold_name"]): float(row["threshold"])
        for _, row in frame.iterrows()
        if str(row["threshold_name"]) in THRESHOLDS
    }
    return ThresholdSet(
        default_0_5=values.get("default_0.5", 0.5),
        sensitivity_90=values["sensitivity_90"],
        youden=values["youden"],
    )


def _load_seed_predictions(seed: int, dataset: str) -> pd.DataFrame:
    logits_path = METHOD_DIR / f"seed_{seed}" / dataset / "logits_calibrated.csv"
    logits_frame = pd.read_csv(logits_path)
    columns = _logit_columns(logits_frame)
    probabilities = _softmax(logits_frame[columns].to_numpy(dtype=float))
    pred = pd.DataFrame(
        {
            "sample_id": logits_frame["sample_id"].astype(str),
            "y_true": logits_frame["y_true"].astype(int),
            "prob_0": probabilities[:, 0],
            "prob_1": probabilities[:, 1],
            "seed": seed,
            "dataset": dataset,
        }
    )
    metadata = _valid_metadata(SPLITS[dataset])
    keep = [
        "sample_id",
        "pid",
        "image_path",
        "label_original",
        "label_cancer_binary",
        "modality",
        "width",
        "height",
    ]
    available = [column for column in keep if column in metadata.columns]
    merged = pred.merge(metadata[available], on="sample_id", how="left")
    merged["patient_id"] = merged["pid"].fillna("unknown").astype(str)
    return merged


def _load_ensemble_predictions(dataset: str) -> tuple[pd.DataFrame, dict[str, float]]:
    frames = [_load_seed_predictions(seed, dataset) for seed in SEEDS]
    base_columns = [
        "sample_id",
        "y_true",
        "patient_id",
        "image_path",
        "label_original",
        "label_cancer_binary",
        "modality",
        "width",
        "height",
    ]
    merged = frames[0][base_columns + ["prob_1"]].rename(columns={"prob_1": "prob_1_seed42"})
    for seed, frame in zip(SEEDS[1:], frames[1:], strict=True):
        merged = merged.merge(
            frame[["sample_id", "prob_1"]].rename(columns={"prob_1": f"prob_1_seed{seed}"}),
            on="sample_id",
            how="inner",
        )
    prob_columns = [f"prob_1_seed{seed}" for seed in SEEDS]
    merged["prob_1"] = merged[prob_columns].mean(axis=1)
    merged["prob_0"] = 1.0 - merged["prob_1"]
    merged["dataset"] = dataset
    merged["seed"] = "ensemble"
    thresholds: dict[str, float] = {"default_0.5": 0.5}
    for threshold_name in ("sensitivity_90", "youden"):
        thresholds[threshold_name] = float(
            np.mean([_read_thresholds(seed).as_dict()[threshold_name] for seed in SEEDS])
        )
    return merged, thresholds


def _binary_confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    labels_bool = labels.astype(int) == 1
    pred_bool = predictions.astype(int) == 1
    return {
        "TP": int(np.sum(labels_bool & pred_bool)),
        "FP": int(np.sum(~labels_bool & pred_bool)),
        "TN": int(np.sum(~labels_bool & ~pred_bool)),
        "FN": int(np.sum(labels_bool & ~pred_bool)),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    labels = labels.astype(int)
    predictions = predictions.astype(int)
    confusion = _binary_confusion(labels, predictions)
    tp = confusion["TP"]
    fp = confusion["FP"]
    tn = confusion["TN"]
    fn = confusion["FN"]
    sensitivity = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    ppv = _safe_divide(tp, tp + fp)
    npv = _safe_divide(tn, tn + fn)
    if not math.isnan(sensitivity) and not math.isnan(specificity):
        balanced_accuracy = (sensitivity + specificity) / 2.0
    else:
        balanced_accuracy = float("nan")
    if len(np.unique(labels)) > 1:
        auroc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))
    else:
        auroc = float("nan")
        auprc = float("nan")
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "balanced_accuracy": balanced_accuracy,
        "macro_F1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "predicted_positive_rate": float(np.mean(predictions)) if len(predictions) else float("nan"),
        **{key: float(value) for key, value in confusion.items()},
    }


def _bootstrap_all_metric_values(
    frame: pd.DataFrame,
    *,
    labels_column: str,
    scores_column: str,
    predictions_column: str,
    cluster_column: str,
    n_bootstrap: int,
    seed: int,
    metrics: Iterable[str] = METRICS,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    metric_names = list(metrics)
    clusters: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _, group in frame.groupby(cluster_column, dropna=False):
        clusters.append(
            (
                group[labels_column].to_numpy(dtype=int),
                group[scores_column].to_numpy(dtype=float),
                group[predictions_column].to_numpy(dtype=int),
            )
        )
    values: dict[str, list[float]] = {metric: [] for metric in metric_names}
    for _ in range(n_bootstrap):
        sampled_indices = rng.integers(0, len(clusters), size=len(clusters))
        labels = np.concatenate([clusters[index][0] for index in sampled_indices])
        scores = np.concatenate([clusters[index][1] for index in sampled_indices])
        predictions = np.concatenate([clusters[index][2] for index in sampled_indices])
        try:
            computed = _classification_metrics(labels, scores, predictions)
        except ValueError:
            continue
        for metric in metric_names:
            value = computed[metric]
            if not math.isnan(float(value)):
                values[metric].append(float(value))
    return values


def _ci(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[~np.isnan(array)]
    if len(array) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "ci_low": float(np.quantile(array, 0.025)),
        "ci_high": float(np.quantile(array, 0.975)),
    }


def run_patient_clustered_bootstrap(n_bootstrap: int) -> None:
    by_seed_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    for dataset in ("cystods_internal", "ebtc_external"):
        for seed in SEEDS:
            frame = _load_seed_predictions(seed, dataset)
            thresholds = _read_thresholds(seed).as_dict()
            n_patients = int(frame["patient_id"].nunique())
            patient_labels = frame.groupby("patient_id")["y_true"].max()
            for threshold_name, threshold in thresholds.items():
                working = frame.copy()
                working["y_pred"] = (working["prob_1"] >= threshold).astype(int)
                point_metrics = _classification_metrics(
                    working["y_true"].to_numpy(dtype=int),
                    working["prob_1"].to_numpy(dtype=float),
                    working["y_pred"].to_numpy(dtype=int),
                )
                bootstrap_by_metric = _bootstrap_all_metric_values(
                    working,
                    labels_column="y_true",
                    scores_column="prob_1",
                    predictions_column="y_pred",
                    cluster_column="patient_id",
                    n_bootstrap=n_bootstrap,
                    seed=10_000 + seed + len(threshold_name),
                )
                for metric in METRICS:
                    values = bootstrap_by_metric[metric]
                    for index, value in enumerate(values):
                        replicate_rows.append(
                            {
                                "dataset": dataset,
                                "evaluation_level": "patient_clustered_image",
                                "threshold_name": threshold_name,
                                "metric": metric,
                                "seed": seed,
                                "bootstrap_index": index,
                                "value": value,
                            }
                        )
                    ci = _ci(values)
                    by_seed_rows.append(
                        {
                            "dataset": dataset,
                            "evaluation_level": "patient_clustered_image",
                            "threshold_name": threshold_name,
                            "metric": metric,
                            "mean": point_metrics[metric],
                            "bootstrap_mean": ci["mean"],
                            "ci_low": ci["ci_low"],
                            "ci_high": ci["ci_high"],
                            "n_bootstrap": len(values),
                            "n_patients": n_patients,
                            "n_images": int(len(working)),
                            "n_positive_patients": int((patient_labels == 1).sum()),
                            "n_negative_patients": int((patient_labels == 0).sum()),
                            "seed": seed,
                        }
                    )

    by_seed = pd.DataFrame(by_seed_rows)
    replicates = pd.DataFrame(replicate_rows)
    summary_rows: list[dict[str, Any]] = []
    group_columns = ["dataset", "evaluation_level", "threshold_name", "metric"]
    for keys, group in replicates.groupby(group_columns, dropna=False):
        key_map = dict(zip(group_columns, keys, strict=True))
        ci = _ci(group["value"].to_numpy(dtype=float))
        source = by_seed
        for column, value in key_map.items():
            source = source[source[column].eq(value)]
        summary_rows.append(
            {
                **key_map,
                "mean": float(source["mean"].mean()),
                "bootstrap_mean": ci["mean"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "n_bootstrap": int(len(group)),
                "n_patients": int(source["n_patients"].iloc[0]),
                "n_images": int(source["n_images"].iloc[0]),
                "n_positive_patients": int(source["n_positive_patients"].iloc[0]),
                "n_negative_patients": int(source["n_negative_patients"].iloc[0]),
                "seed": "summary",
            }
        )

    PATIENT_OUT.mkdir(parents=True, exist_ok=True)
    write_table(by_seed, PATIENT_OUT / "patient_clustered_bootstrap_by_seed.csv")
    write_table(pd.DataFrame(summary_rows), PATIENT_OUT / "patient_clustered_bootstrap_summary.csv")
    write_table(replicates, PATIENT_OUT / "patient_clustered_bootstrap_replicates.csv")


def _patient_score(frame: pd.DataFrame, aggregation_rule: str, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for patient_id, group in frame.groupby("patient_id", dropna=False):
        probs = np.sort(group["prob_1"].to_numpy(dtype=float))[::-1]
        labels = group["y_true"].to_numpy(dtype=int)
        if aggregation_rule == "max_probability":
            score = float(probs[0])
            predicted = int(score >= threshold)
        elif aggregation_rule == "top3_mean":
            score = float(np.mean(probs[: min(3, len(probs))]))
            predicted = int(score >= threshold)
        elif aggregation_rule == "top5_mean":
            score = float(np.mean(probs[: min(5, len(probs))]))
            predicted = int(score >= threshold)
        elif aggregation_rule == "mean_probability":
            score = float(np.mean(probs))
            predicted = int(score >= threshold)
        elif aggregation_rule == "positive_image_proportion":
            score = float(np.mean(group["prob_1"].to_numpy(dtype=float) >= threshold))
            predicted = int(score > 0.0)
        else:
            msg = f"Unknown aggregation rule: {aggregation_rule}"
            raise ValueError(msg)
        rows.append(
            {
                "patient_id": str(patient_id),
                "n_images": int(len(group)),
                "n_positive_images_ground_truth": int(labels.sum()),
                "patient_true_label": int(labels.max()),
                "patient_score": score,
                "patient_predicted_label": predicted,
            }
        )
    return pd.DataFrame(rows)


def _patient_aggregation_metrics(
    frame: pd.DataFrame,
    *,
    dataset: str,
    seed: str | int,
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for aggregation_rule in PATIENT_AGGREGATIONS:
        for threshold_name, threshold in thresholds.items():
            patient_frame = _patient_score(frame, aggregation_rule, threshold)
            labels = patient_frame["patient_true_label"].to_numpy(dtype=int)
            scores = patient_frame["patient_score"].to_numpy(dtype=float)
            preds = patient_frame["patient_predicted_label"].to_numpy(dtype=int)
            metrics = _classification_metrics(labels, scores, preds)
            row = {
                "dataset": dataset,
                "split": dataset,
                "aggregation_rule": aggregation_rule,
                "threshold_rule": threshold_name,
                "threshold_value": threshold,
                "seed": seed,
                "n_patients": int(len(patient_frame)),
                "n_positive_patients": int(labels.sum()),
                "n_negative_patients": int((labels == 0).sum()),
                **metrics,
            }
            rows.append(row)
            confusion_rows.append(
                {
                    "dataset": dataset,
                    "aggregation_rule": aggregation_rule,
                    "threshold_rule": threshold_name,
                    "seed": seed,
                    "TP": metrics["TP"],
                    "FP": metrics["FP"],
                    "TN": metrics["TN"],
                    "FN": metrics["FN"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(confusion_rows)


def _patient_predictions_table(
    frame: pd.DataFrame,
    *,
    dataset: str,
    seed: str | int,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for patient_id, group in frame.groupby("patient_id", dropna=False):
        probs = np.sort(group["prob_1"].to_numpy(dtype=float))[::-1]
        labels = group["y_true"].to_numpy(dtype=int)
        max_probability = float(probs[0])
        top3 = float(np.mean(probs[: min(3, len(probs))]))
        top5 = float(np.mean(probs[: min(5, len(probs))]))
        mean_probability = float(np.mean(probs))
        patient_true = int(labels.max())
        prop_default = float(np.mean(group["prob_1"] >= thresholds["default_0.5"]))
        prop_sens90 = float(np.mean(group["prob_1"] >= thresholds["sensitivity_90"]))
        pred_max = int(max_probability >= thresholds["default_0.5"])
        pred_top3 = int(top3 >= thresholds["default_0.5"])
        pred_prop = int(prop_default > 0.0)
        if pred_max == 1 and patient_true == 1:
            error_type = "TP"
        elif pred_max == 1 and patient_true == 0:
            error_type = "FP"
        elif pred_max == 0 and patient_true == 0:
            error_type = "TN"
        else:
            error_type = "FN"
        rows.append(
            {
                "dataset": dataset,
                "patient_id": str(patient_id),
                "seed_or_ensemble": seed,
                "n_images": int(len(group)),
                "n_positive_images_ground_truth": int(labels.sum()),
                "patient_true_label": patient_true,
                "max_probability": max_probability,
                "top3_mean_probability": top3,
                "top5_mean_probability": top5,
                "mean_probability": mean_probability,
                "positive_image_proportion_default": prop_default,
                "positive_image_proportion_sensitivity90": prop_sens90,
                "predicted_label_max": pred_max,
                "predicted_label_top3": pred_top3,
                "predicted_label_proportion": pred_prop,
                "patient_error_type": error_type,
            }
        )
    return pd.DataFrame(rows)


def run_patient_aggregation(n_bootstrap: int) -> None:
    by_seed_rows: list[pd.DataFrame] = []
    confusion_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    ci_rows: list[dict[str, Any]] = []
    for dataset in ("cystods_internal", "ebtc_external"):
        for seed in SEEDS:
            frame = _load_seed_predictions(seed, dataset)
            thresholds = _read_thresholds(seed).as_dict()
            metrics, confusion = _patient_aggregation_metrics(
                frame,
                dataset=dataset,
                seed=seed,
                thresholds=thresholds,
            )
            by_seed_rows.append(metrics)
            confusion_rows.append(confusion)
            prediction_rows.append(
                _patient_predictions_table(frame, dataset=dataset, seed=seed, thresholds=thresholds)
            )
            for _, row in metrics.iterrows():
                patient_frame = _patient_score(
                    frame,
                    str(row["aggregation_rule"]),
                    float(row["threshold_value"]),
                )
                working = patient_frame.rename(
                    columns={
                        "patient_true_label": "label",
                        "patient_score": "score",
                        "patient_predicted_label": "pred",
                    }
                )
                bootstrap_by_metric = _bootstrap_all_metric_values(
                    working,
                    labels_column="label",
                    scores_column="score",
                    predictions_column="pred",
                    cluster_column="patient_id",
                    n_bootstrap=n_bootstrap,
                    seed=30_000 + seed + int(row.name),
                )
                for metric in METRICS:
                    values = bootstrap_by_metric[metric]
                    ci = _ci(values)
                    ci_rows.append(
                        {
                            "dataset": dataset,
                            "aggregation_rule": row["aggregation_rule"],
                            "threshold_rule": row["threshold_rule"],
                            "metric": metric,
                            "mean": row[metric],
                            "ci_low": ci["ci_low"],
                            "ci_high": ci["ci_high"],
                            "n_bootstrap": len(values),
                            "n_patients": row["n_patients"],
                            "seed": seed,
                        }
                    )

        ensemble_frame, ensemble_thresholds = _load_ensemble_predictions(dataset)
        ensemble_metrics, ensemble_confusion = _patient_aggregation_metrics(
            ensemble_frame,
            dataset=dataset,
            seed="ensemble",
            thresholds=ensemble_thresholds,
        )
        by_seed_rows.append(ensemble_metrics)
        confusion_rows.append(ensemble_confusion)
        prediction_rows.append(
            _patient_predictions_table(
                ensemble_frame,
                dataset=dataset,
                seed="ensemble",
                thresholds=ensemble_thresholds,
            )
        )

    by_seed = pd.concat(by_seed_rows, ignore_index=True)
    confusion_all = pd.concat(confusion_rows, ignore_index=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    seed_only = by_seed[by_seed["seed"].astype(str).ne("ensemble")].copy()
    summary_metrics = [
        "AUROC",
        "AUPRC",
        "balanced_accuracy",
        "macro_F1",
        "sensitivity",
        "specificity",
        "PPV",
        "NPV",
        "predicted_positive_rate",
        "TP",
        "FP",
        "TN",
        "FN",
    ]
    summary = (
        seed_only.groupby(
            [
                "dataset",
                "split",
                "aggregation_rule",
                "threshold_rule",
                "n_patients",
                "n_positive_patients",
                "n_negative_patients",
            ],
            dropna=False,
        )[summary_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in column if part])
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    write_table(by_seed, PATIENT_OUT / "patient_level_aggregation_by_seed.csv")
    write_table(summary, PATIENT_OUT / "patient_level_aggregation_summary.csv")
    write_table(pd.DataFrame(ci_rows), PATIENT_OUT / "patient_level_aggregation_bootstrap_ci.csv")
    write_table(confusion_all, PATIENT_OUT / "patient_level_confusion_matrices.csv")
    write_table(
        predictions[predictions["seed_or_ensemble"].astype(str).ne("ensemble")],
        PATIENT_OUT / "patient_level_predictions_by_seed.csv",
    )
    write_table(
        predictions[predictions["seed_or_ensemble"].astype(str).eq("ensemble")],
        PATIENT_OUT / "patient_level_predictions_ensemble.csv",
    )


def _adjusted_ppv_npv(sensitivity: float, specificity: float, prevalence: float) -> tuple[float, float]:
    ppv_denom = sensitivity * prevalence + (1.0 - specificity) * (1.0 - prevalence)
    npv_denom = (1.0 - sensitivity) * prevalence + specificity * (1.0 - prevalence)
    ppv = _safe_divide(sensitivity * prevalence, ppv_denom)
    npv = _safe_divide(specificity * (1.0 - prevalence), npv_denom)
    return ppv, npv


def run_prevalence_reporting() -> None:
    rows: list[dict[str, Any]] = []
    for dataset in ("cystods_internal", "ebtc_external"):
        for seed in SEEDS:
            frame = _load_seed_predictions(seed, dataset)
            thresholds = _read_thresholds(seed).as_dict()
            labels = frame["y_true"].to_numpy(dtype=int)
            scores = frame["prob_1"].to_numpy(dtype=float)
            observed_prevalence = float(np.mean(labels))
            observed_auprc = float(average_precision_score(labels, scores))
            normalized_gain = (observed_auprc - observed_prevalence) / (1.0 - observed_prevalence)
            assumed_values = [*ASSUMED_PREVALENCES, observed_prevalence]
            for threshold_name, threshold in thresholds.items():
                preds = (scores >= threshold).astype(int)
                metrics = _classification_metrics(labels, scores, preds)
                for assumed_prevalence in assumed_values:
                    ppv, npv = _adjusted_ppv_npv(
                        metrics["sensitivity"],
                        metrics["specificity"],
                        float(assumed_prevalence),
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "threshold_name": threshold_name,
                            "positive_prevalence_observed": observed_prevalence,
                            "no_skill_AUPRC": observed_prevalence,
                            "observed_AUPRC": observed_auprc,
                            "normalized_AUPRC_gain": normalized_gain,
                            "sensitivity": metrics["sensitivity"],
                            "specificity": metrics["specificity"],
                            "assumed_prevalence": float(assumed_prevalence),
                            "prevalence_adjusted_PPV": ppv,
                            "prevalence_adjusted_NPV": npv,
                        }
                    )
    by_seed = pd.DataFrame(rows)
    group_columns = ["dataset", "threshold_name", "assumed_prevalence"]
    value_columns = [
        "positive_prevalence_observed",
        "no_skill_AUPRC",
        "observed_AUPRC",
        "normalized_AUPRC_gain",
        "sensitivity",
        "specificity",
        "prevalence_adjusted_PPV",
        "prevalence_adjusted_NPV",
    ]
    summary = by_seed.groupby(group_columns, dropna=False)[value_columns].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join([str(part) for part in column if part])
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    PREVALENCE_OUT.mkdir(parents=True, exist_ok=True)
    write_table(by_seed, PREVALENCE_OUT / "prevalence_aware_ppv_npv_by_seed.csv")
    write_table(summary, PREVALENCE_OUT / "prevalence_aware_ppv_npv_summary.csv")


def _threshold_sweep(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.0, 1.0, 1001):
        preds = (scores >= threshold).astype(int)
        metrics = _classification_metrics(labels, scores, preds)
        rows.append({"threshold": float(threshold), **metrics})
    return pd.DataFrame(rows)


def _select_modality_thresholds(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    sweep = _threshold_sweep(labels, scores)
    eligible = sweep[sweep["sensitivity"] >= 0.90].copy()
    if eligible.empty:
        sensitivity_90 = float(sweep.sort_values("sensitivity", ascending=False)["threshold"].iloc[0])
    else:
        sensitivity_90 = float(
            eligible.sort_values(["specificity", "threshold"], ascending=[False, False])[
                "threshold"
            ].iloc[0]
        )
    sweep["youden"] = sweep["sensitivity"] + sweep["specificity"] - 1.0
    youden = float(sweep.sort_values(["youden", "threshold"], ascending=[False, False])["threshold"].iloc[0])
    return {"modality_specific_sensitivity_90": sensitivity_90, "modality_specific_youden": youden}


def run_modality_specific_thresholds() -> None:
    rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        calib = _load_seed_predictions(seed, "cystods_calib")
        internal = _load_seed_predictions(seed, "cystods_internal")
        shared = _read_thresholds(seed).as_dict()
        for modality, calib_group in calib.groupby("modality", dropna=False):
            modality = str(modality)
            modality_thresholds = _select_modality_thresholds(
                calib_group["y_true"].to_numpy(dtype=int),
                calib_group["prob_1"].to_numpy(dtype=float),
            )
            for rule_name, threshold in {**{f"shared_{k}": v for k, v in shared.items()}, **modality_thresholds}.items():
                threshold_rows.append(
                    {
                        "seed": seed,
                        "modality": modality,
                        "threshold_rule": rule_name,
                        "threshold_value": threshold,
                        "selected_on": "cystods_calib",
                    }
                )
                test_group = internal[internal["modality"].astype(str).eq(modality)].copy()
                labels = test_group["y_true"].to_numpy(dtype=int)
                scores = test_group["prob_1"].to_numpy(dtype=float)
                preds = (scores >= threshold).astype(int)
                metrics = _classification_metrics(labels, scores, preds)
                negatives = metrics["TN"] + metrics["FP"]
                rows.append(
                    {
                        "modality": modality,
                        "threshold_rule": rule_name,
                        "threshold_value": threshold,
                        "dataset": "cystods_internal",
                        "seed": seed,
                        "sensitivity": metrics["sensitivity"],
                        "specificity": metrics["specificity"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "macro_F1": metrics["macro_F1"],
                        "PPV": metrics["PPV"],
                        "NPV": metrics["NPV"],
                        "predicted_positive_rate": metrics["predicted_positive_rate"],
                        "false_alarms_per_100_negative_images": _safe_divide(
                            100.0 * metrics["FP"],
                            negatives,
                        ),
                        "TP": metrics["TP"],
                        "FP": metrics["FP"],
                        "TN": metrics["TN"],
                        "FN": metrics["FN"],
                    }
                )
    by_seed = pd.DataFrame(rows)
    summary = (
        by_seed.groupby(["modality", "threshold_rule", "dataset"], dropna=False)[
            [
                "threshold_value",
                "sensitivity",
                "specificity",
                "balanced_accuracy",
                "macro_F1",
                "PPV",
                "NPV",
                "predicted_positive_rate",
                "false_alarms_per_100_negative_images",
                "TP",
                "FP",
                "TN",
                "FN",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in column if part])
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    write_table(by_seed, Path("outputs/v2/modality_analysis/modality_specific_threshold_by_seed.csv"))
    write_table(summary, Path("outputs/v2/modality_analysis/modality_specific_threshold_summary.csv"))
    write_table(pd.DataFrame(threshold_rows), Path("outputs/v2/modality_analysis/modality_thresholds_selected_on_calib.csv"))


def run_probability_distribution() -> None:
    frame, _ = _load_ensemble_predictions("ebtc_external")
    out = frame[
        [
            "dataset",
            "sample_id",
            "patient_id",
            "label_original",
            "label_cancer_binary",
            "modality",
            "prob_1",
        ]
    ].rename(columns={"prob_1": "ensemble_cancer_probability"})
    DIAGNOSTIC_OUT.mkdir(parents=True, exist_ok=True)
    write_table(out, DIAGNOSTIC_OUT / "ebtc_probability_by_original_label.csv")
    summary = (
        out.groupby("label_original", dropna=False)["ensemble_cancer_probability"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    write_table(summary, DIAGNOSTIC_OUT / "ebtc_probability_by_original_label_summary.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    bins = np.linspace(0.0, 1.0, 41)
    colors = {"HGC": "#b84a62", "LGC": "#d9895b", "NTL": "#4477aa", "NST": "#3f8f72"}
    for label in ("HGC", "LGC", "NTL", "NST"):
        values = out[out["label_original"].astype(str).eq(label)]["ensemble_cancer_probability"]
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=label,
            color=colors[label],
        )
    ax.set_xlabel("Ensemble cancer probability")
    ax.set_ylabel("Density")
    ax.set_title("Cancer probability distribution by EBTC original label")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    figure_path = Path("outputs/v2/figures/ebtc_probability_density_by_original_label.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)


def _target_layers(model: Any) -> list[Any]:
    if hasattr(model, "layers"):
        layers = model.layers
        last = layers[-1]
        if hasattr(last, "blocks"):
            block = last.blocks[-1]
            return [getattr(block, "norm1", block)]
        return [last]
    if hasattr(model, "stages"):
        stage = model.stages[-1]
        if hasattr(stage, "blocks"):
            return [stage.blocks[-1]]
        return [stage]
    for name in ("conv_head", "layer4"):
        if hasattr(model, name):
            layer = getattr(model, name)
            if name == "layer4" and hasattr(layer, "__getitem__"):
                return [layer[-1]]
            return [layer]
    msg = "Could not infer Grad-CAM target layer for model."
    raise AttributeError(msg)


def _reshape_tokens(tensor: Any) -> Any:
    if tensor.ndim != 3:
        return tensor
    batch, tokens, channels = tensor.shape
    height = int(math.sqrt(tokens))
    if height * height != tokens and int(math.sqrt(tokens - 1)) ** 2 == tokens - 1:
        tensor = tensor[:, 1:, :]
        tokens -= 1
        height = int(math.sqrt(tokens))
    width = tokens // height
    return tensor.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()


def _load_visual_model(checkpoint_path: Path) -> tuple[Any, dict[str, Any], str]:
    device = get_device(prefer_cuda=True)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model_config = dict(config.get("model", {}) or {})
    model_config["num_classes"] = int((config.get("task", {}) or {}).get("num_classes", 2))
    model = create_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model, config, device


def _resolve_image_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    return ROOT / path


def _select_cases() -> pd.DataFrame:
    ebtc, _ = _load_ensemble_predictions("ebtc_external")
    cystods, _ = _load_ensemble_predictions("cystods_internal")
    ebtc["pred_default"] = (ebtc["prob_1"] >= 0.5).astype(int)
    cystods["pred_default"] = (cystods["prob_1"] >= 0.5).astype(int)
    case_specs = [
        ("HGC true positive", ebtc, lambda f: (f["label_original"] == "HGC") & (f["pred_default"] == 1), False),
        ("LGC true positive", ebtc, lambda f: (f["label_original"] == "LGC") & (f["pred_default"] == 1), False),
        ("HGC false negative", ebtc, lambda f: (f["label_original"] == "HGC") & (f["pred_default"] == 0), True),
        ("LGC false negative", ebtc, lambda f: (f["label_original"] == "LGC") & (f["pred_default"] == 0), True),
        ("NTL false positive", ebtc, lambda f: (f["label_original"] == "NTL") & (f["pred_default"] == 1), False),
        ("NST true negative", ebtc, lambda f: (f["label_original"] == "NST") & (f["pred_default"] == 0), True),
        ("BLC false positive", cystods, lambda f: (f["modality"] == "BLC") & (f["y_true"] == 0) & (f["pred_default"] == 1), False),
        ("WLC true negative", cystods, lambda f: (f["modality"] == "WLC") & (f["y_true"] == 0) & (f["pred_default"] == 0), True),
    ]
    rows: list[pd.DataFrame] = []
    for error_type, frame, selector, ascending in case_specs:
        selected = frame[selector(frame)].copy()
        selected = selected.sort_values("prob_1", ascending=ascending).head(2)
        selected["error_type"] = error_type
        rows.append(selected)
    cases = pd.concat(rows, ignore_index=True)
    cases["case_index"] = np.arange(1, len(cases) + 1)
    return cases


def run_visual_audit(max_cases: int | None = None) -> None:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    selected_runs = pd.read_csv("outputs/v2/final_validation/frozen_baselines/selected_frozen_runs.csv")
    row = selected_runs[(selected_runs["method_id"] == "B3") & (selected_runs["seed"].astype(int) == 42)].iloc[0]
    checkpoint = Path(str(row["run_dir"])) / "checkpoint_best.pt"
    model, config, device = _load_visual_model(checkpoint)
    transform = _build_transform(config, train=False)
    cam = GradCAM(model=model, target_layers=_target_layers(model), reshape_transform=_reshape_tokens)

    cases = _select_cases()
    if max_cases is not None:
        cases = cases.head(max_cases).copy()
    heatmap_dir = VISUAL_OUT / "heatmaps"
    overlay_dir = VISUAL_OUT / "overlays"
    panel_dir = VISUAL_OUT / "panels"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for _, row in cases.iterrows():
        image_path = _resolve_image_path(str(row["image_path"]))
        metadata = pd.DataFrame([row.to_dict()])
        dataset = V2MetadataDataset(metadata, label_column="y_true", transform=transform)
        item = dataset[0]
        batch = collate_v2([item])
        grayscale = cam(
            input_tensor=batch["image"].to(device),
            targets=[ClassifierOutputTarget(1)],
        )[0]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        resized_cam = cv2.resize(grayscale, (image.shape[1], image.shape[0]))
        overlay = overlay_cam_on_image(image, resized_cam)
        case_slug = f"{int(row['case_index']):02d}_{str(row['error_type']).replace(' ', '_')}_{row['sample_id']}"
        heatmap_path = heatmap_dir / f"{case_slug}.png"
        overlay_path = overlay_dir / f"{case_slug}.png"
        panel_path = panel_dir / f"{case_slug}.png"
        plt.imsave(heatmap_path, resized_cam, cmap="jet")
        Image.fromarray(overlay).save(overlay_path)
        save_cam_case_figure(
            image,
            resized_cam,
            None,
            panel_path,
            title=f"{row['error_type']} | p={float(row['prob_1']):.3f}",
        )
        rows.append(
            {
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "patient_id": row["patient_id"],
                "original_label": row.get("label_original", ""),
                "binary_label": int(row["y_true"]),
                "modality": row.get("modality", ""),
                "threshold_name": "default_0.5",
                "predicted_probability": float(row["prob_1"]),
                "predicted_label": int(row["pred_default"]),
                "error_type": row["error_type"],
                "image_path": str(image_path),
                "heatmap_path": str(heatmap_path),
                "overlay_path": str(overlay_path),
                "panel_path": str(panel_path),
                "seed_or_ensemble": "ensemble_selected_seed42_cam",
                "cam_target_class": "cancer",
            }
        )
    VISUAL_OUT.mkdir(parents=True, exist_ok=True)
    write_table(pd.DataFrame(rows), VISUAL_OUT / "visual_audit_cases.csv")
    (VISUAL_OUT / "visual_audit_note.md").write_text(
        "# CDG-Cysto v2 Visual Audit\n\n"
        "Cases were selected using ensemble mean calibrated cancer probability. "
        "Grad-CAM overlays were generated with the seed-42 frozen checkpoint as a "
        "qualitative visual audit, targeting the cancer logit for all cases.\n",
        encoding="utf-8",
    )


def run_source_tables() -> None:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2("outputs/v2/tables/table1_dataset_split_statistics.csv", SOURCE_OUT / "fig1_workflow_values.csv")
    shutil.copy2("outputs/v2/tables/table1_dataset_split_statistics.csv", SOURCE_OUT / "fig2_dataset_structure.csv")
    shutil.copy2(
        "outputs/v2/final_validation/frozen_baselines/frozen_baseline_comparison.csv",
        SOURCE_OUT / "fig3_frozen_validation_source.csv",
    )
    threshold_source = pd.read_csv("outputs/v2/operating_points/named_operating_points_summary.csv")
    calibration_source = pd.read_csv("outputs/v2/final_validation/frozen_baselines/calibration_before_after.csv")
    threshold_source.to_csv(SOURCE_OUT / "fig4_threshold_source.csv", index=False)
    calibration_source.to_csv(SOURCE_OUT / "fig4_calibration_source.csv", index=False)
    if (VISUAL_OUT / "visual_audit_cases.csv").exists():
        shutil.copy2(VISUAL_OUT / "visual_audit_cases.csv", SOURCE_OUT / "fig5_failure_visual_audit_source.csv")
    if (PATIENT_OUT / "patient_level_aggregation_summary.csv").exists():
        shutil.copy2(
            PATIENT_OUT / "patient_level_aggregation_summary.csv",
            SOURCE_OUT / "fig6_patient_level_aggregation_source.csv",
        )


def write_manifest(args: argparse.Namespace) -> None:
    manifest = {
        "patient_clustered_bootstrap": str(PATIENT_OUT / "patient_clustered_bootstrap_summary.csv"),
        "patient_level_aggregation": str(PATIENT_OUT / "patient_level_aggregation_summary.csv"),
        "patient_level_predictions": str(PATIENT_OUT / "patient_level_predictions_ensemble.csv"),
        "prevalence_aware_reporting": str(PREVALENCE_OUT / "prevalence_aware_ppv_npv_summary.csv"),
        "visual_audit": str(VISUAL_OUT / "visual_audit_cases.csv"),
        "modality_specific_thresholds": str(
            Path("outputs/v2/modality_analysis/modality_specific_threshold_summary.csv")
        ),
        "probability_density": str(Path("outputs/v2/figures/ebtc_probability_density_by_original_label.png")),
        "n_bootstrap": args.n_bootstrap,
        "visual_audit_mode": "seed42_gradcam_qualitative",
    }
    write_json(manifest, Path("outputs/v2/reports/patient_prevalence_visual_audit_manifest.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--max-visual-cases", type=int)
    args = parser.parse_args()

    PATIENT_OUT.mkdir(parents=True, exist_ok=True)
    PREVALENCE_OUT.mkdir(parents=True, exist_ok=True)
    VISUAL_OUT.mkdir(parents=True, exist_ok=True)

    run_patient_clustered_bootstrap(args.n_bootstrap)
    run_patient_aggregation(args.n_bootstrap)
    run_prevalence_reporting()
    run_modality_specific_thresholds()
    run_probability_distribution()
    if not args.skip_visual:
        run_visual_audit(max_cases=args.max_visual_cases)
    run_source_tables()
    write_manifest(args)
    print("Wrote patient-level, prevalence-aware, modality-threshold, and visual-audit outputs.")


if __name__ == "__main__":
    main()
