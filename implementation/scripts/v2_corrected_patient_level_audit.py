"""Corrected EBTC case-level patient audit and aggregation sensitivity analysis."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdg_cysto.utils.io import write_table  # noqa: E402

SEEDS = (42, 2025, 3407)
DATASETS = ("cystods_internal", "ebtc_external")
THRESHOLDS = ("default_0.5", "sensitivity_90", "youden")
AGGREGATIONS = (
    "max_probability",
    "top3_mean",
    "top5_mean",
    "mean_probability",
    "positive_image_proportion",
)

METHOD_DIR = Path("outputs/v2/final_validation/frozen_baselines/B3_CDG_Cysto_v2")
SPLITS = {
    "cystods_calib": Path("outputs/v2/splits/cystods_calib.csv"),
    "cystods_internal": Path("outputs/v2/splits/cystods_test_internal.csv"),
    "ebtc_external": Path("outputs/v2/splits/ebtc_external_all.csv"),
}
OUT = Path("outputs/v2/patient_level_corrected")


def normalize_ebtc_case(value: str) -> str:
    """Normalize EBTC case number from canonical and legacy image names."""

    match = re.search(r"case_0*(\d+)", str(value), flags=re.IGNORECASE)
    if match:
        return f"case_{int(match.group(1)):03d}"
    return "case_unknown"


def normalize_ebtc_case_pt(value: str) -> str:
    case_id = normalize_ebtc_case(value)
    text = str(value)
    for pattern in (r"pt_0*(\d+)", r"pt0*(\d+)", r"cys_pt0*(\d+)"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return f"{case_id}_pt_{int(match.group(1)):03d}"
    return f"{case_id}_pt_unknown"


def binary_confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    labels_bool = labels.astype(int) == 1
    pred_bool = predictions.astype(int) == 1
    return {
        "TP": int(np.sum(labels_bool & pred_bool)),
        "FP": int(np.sum(~labels_bool & pred_bool)),
        "TN": int(np.sum(~labels_bool & ~pred_bool)),
        "FN": int(np.sum(labels_bool & ~pred_bool)),
    }


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else math.nan


def safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(labels, scores)) if np.sum(labels) > 0 else math.nan


def classification_metrics(labels: np.ndarray, scores: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    confusion = binary_confusion(labels, preds)
    tp = confusion["TP"]
    fp = confusion["FP"]
    tn = confusion["TN"]
    fn = confusion["FN"]
    sensitivity = tp / (tp + fn) if (tp + fn) else math.nan
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    ppv = tp / (tp + fp) if (tp + fp) else math.nan
    npv = tn / (tn + fn) if (tn + fn) else math.nan
    return {
        "AUROC": safe_auc(labels, scores),
        "AUPRC": safe_auprc(labels, scores),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds))
        if len(np.unique(labels)) == 2
        else math.nan,
        "macro_F1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "PPV": float(ppv),
        "NPV": float(npv),
        "predicted_positive_rate": float(np.mean(preds)),
        **{key: float(value) for key, value in confusion.items()},
    }


def read_image_thresholds(seed: int) -> dict[str, float]:
    frame = pd.read_csv(METHOD_DIR / f"seed_{seed}" / "threshold_metrics.csv")
    thresholds = {"default_0.5": 0.5}
    for _, row in frame.iterrows():
        name = str(row["threshold_name"])
        if name in THRESHOLDS:
            thresholds[name] = float(row["threshold"])
    return thresholds


def load_predictions(seed: int, dataset: str, *, grouping: str) -> pd.DataFrame:
    pred = pd.read_csv(
        METHOD_DIR / f"seed_{seed}" / dataset / "predictions_default_0.5.csv"
    )
    metadata = pd.read_csv(SPLITS[dataset])
    keep = [
        "sample_id",
        "pid",
        "visit",
        "label_original",
        "label_cancer_binary",
        "modality",
        "image_path",
    ]
    merged = pred.merge(metadata[[c for c in keep if c in metadata.columns]], on="sample_id", how="left")
    if dataset == "ebtc_external":
        if grouping == "case_level":
            merged["patient_id"] = merged["sample_id"].map(normalize_ebtc_case)
        elif grouping == "case_pt_level":
            merged["patient_id"] = merged["sample_id"].map(normalize_ebtc_case_pt)
        else:
            merged["patient_id"] = merged["pid"].fillna("unknown").astype(str)
    else:
        merged["patient_id"] = merged["pid"].astype(str)
    merged["seed"] = seed
    merged["dataset"] = dataset
    merged["grouping_policy"] = grouping if dataset == "ebtc_external" else "official_pid"
    return merged


def load_ensemble(dataset: str, *, grouping: str) -> pd.DataFrame:
    frames = [load_predictions(seed, dataset, grouping=grouping) for seed in SEEDS]
    base_cols = [
        "sample_id",
        "y_true",
        "patient_id",
        "image_path",
        "label_original",
        "label_cancer_binary",
        "modality",
        "dataset",
        "grouping_policy",
    ]
    merged = frames[0][base_cols + ["prob_1"]].rename(columns={"prob_1": "prob_1_seed42"})
    for seed, frame in zip(SEEDS[1:], frames[1:], strict=True):
        merged = merged.merge(
            frame[["sample_id", "prob_1"]].rename(columns={"prob_1": f"prob_1_seed{seed}"}),
            on="sample_id",
            how="inner",
        )
    prob_cols = [f"prob_1_seed{seed}" for seed in SEEDS]
    merged["prob_1"] = merged[prob_cols].mean(axis=1)
    merged["prob_0"] = 1.0 - merged["prob_1"]
    merged["seed"] = "ensemble"
    return merged


def patient_labels(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("patient_id")["y_true"].max()


def patient_score(frame: pd.DataFrame, aggregation_rule: str, image_threshold: float = 0.5) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for patient_id, group in frame.groupby("patient_id", dropna=False):
        probs = np.sort(group["prob_1"].to_numpy(dtype=float))[::-1]
        labels = group["y_true"].to_numpy(dtype=int)
        if aggregation_rule == "max_probability":
            score = float(probs[0])
        elif aggregation_rule == "top3_mean":
            score = float(np.mean(probs[: min(3, len(probs))]))
        elif aggregation_rule == "top5_mean":
            score = float(np.mean(probs[: min(5, len(probs))]))
        elif aggregation_rule == "mean_probability":
            score = float(np.mean(probs))
        elif aggregation_rule == "positive_image_proportion":
            score = float(np.mean(group["prob_1"].to_numpy(dtype=float) >= image_threshold))
        else:
            raise ValueError(f"Unknown aggregation rule: {aggregation_rule}")
        rows.append(
            {
                "patient_id": str(patient_id),
                "n_images": int(len(group)),
                "n_positive_images_ground_truth": int(labels.sum()),
                "patient_true_label": int(labels.max()),
                "patient_score": score,
            }
        )
    return pd.DataFrame(rows)


def select_threshold(labels: np.ndarray, scores: np.ndarray, rule: str) -> float:
    if rule == "default_0.5":
        return 0.5
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    best_threshold = 0.5
    best_value = -np.inf
    for threshold in candidates:
        preds = (scores >= threshold).astype(int)
        metrics = classification_metrics(labels, scores, preds)
        if rule == "youden":
            value = metrics["sensitivity"] + metrics["specificity"] - 1.0
        elif rule == "sensitivity_90":
            if metrics["sensitivity"] < 0.90:
                continue
            value = metrics["specificity"]
        else:
            raise ValueError(f"Unknown threshold rule: {rule}")
        if value > best_value or (math.isclose(value, best_value) and threshold > best_threshold):
            best_value = value
            best_threshold = float(threshold)
    return float(best_threshold)


def summarize_grouping() -> None:
    metadata = pd.read_csv(SPLITS["ebtc_external"])
    metadata["current_pid"] = metadata["pid"].fillna("unknown").astype(str)
    metadata["case_level_patient_id"] = metadata["sample_id"].map(normalize_ebtc_case)
    metadata["case_pt_level_patient_id"] = metadata["sample_id"].map(normalize_ebtc_case_pt)
    rows = []
    for policy, column in [
        ("current_pid", "current_pid"),
        ("case_level", "case_level_patient_id"),
        ("case_pt_level", "case_pt_level_patient_id"),
    ]:
        labels = metadata.groupby(column)["label_cancer_binary"].max()
        rows.append(
            {
                "grouping_policy": policy,
                "n_groups": int(labels.shape[0]),
                "n_positive_groups": int((labels == 1).sum()),
                "n_negative_groups": int((labels == 0).sum()),
                "n_images": int(len(metadata)),
                "note": "case_level is the preferred corrected EBTC patient proxy"
                if policy == "case_level"
                else "",
            }
        )
    write_table(pd.DataFrame(rows), OUT / "ebtc_patient_grouping_audit.csv")
    case_detail = (
        metadata.groupby("case_level_patient_id")
        .agg(
            n_images=("sample_id", "count"),
            n_positive_images=("label_cancer_binary", "sum"),
            labels=("label_original", lambda s: ",".join(sorted(pd.unique(s)))),
            modalities=("modality", lambda s: ",".join(sorted(pd.unique(s)))),
        )
        .reset_index()
    )
    write_table(case_detail, OUT / "ebtc_case_level_grouping_detail.csv")


def clustered_bootstrap(frame: pd.DataFrame, *, threshold: float, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    patients = np.array(sorted(frame["patient_id"].astype(str).unique()))
    rows = []
    for index in range(n_bootstrap):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        pieces = []
        for draw, patient_id in enumerate(sampled):
            block = frame[frame["patient_id"].astype(str) == patient_id].copy()
            block["_draw"] = draw
            pieces.append(block)
        sampled_frame = pd.concat(pieces, ignore_index=True)
        labels = sampled_frame["y_true"].to_numpy(dtype=int)
        scores = sampled_frame["prob_1"].to_numpy(dtype=float)
        preds = (scores >= threshold).astype(int)
        metrics = classification_metrics(labels, scores, preds)
        rows.extend({"bootstrap_index": index, "metric": k, "value": v} for k, v in metrics.items())
    return pd.DataFrame(rows)


def run_clustered_bootstrap(n_bootstrap: int) -> None:
    by_seed_rows = []
    replicate_rows = []
    for dataset in DATASETS:
        grouping = "case_level" if dataset == "ebtc_external" else "official_pid"
        for seed in SEEDS:
            frame = load_predictions(seed, dataset, grouping=grouping)
            labels_by_patient = patient_labels(frame)
            thresholds = read_image_thresholds(seed)
            for threshold_name, threshold in thresholds.items():
                labels = frame["y_true"].to_numpy(dtype=int)
                scores = frame["prob_1"].to_numpy(dtype=float)
                preds = (scores >= threshold).astype(int)
                point = classification_metrics(labels, scores, preds)
                reps = clustered_bootstrap(
                    frame,
                    threshold=threshold,
                    n_bootstrap=n_bootstrap,
                    seed=100_000 + seed + len(dataset) + len(threshold_name),
                )
                for metric, value in point.items():
                    values = reps[reps["metric"] == metric]["value"].dropna().to_numpy(dtype=float)
                    by_seed_rows.append(
                        {
                            "dataset": dataset,
                            "evaluation_level": "patient_clustered_image",
                            "grouping_policy": grouping,
                            "threshold_policy": "image_level_calibration_threshold",
                            "threshold_name": threshold_name,
                            "threshold_value": threshold,
                            "metric": metric,
                            "mean": value,
                            "bootstrap_mean": float(np.mean(values)) if len(values) else math.nan,
                            "ci_low": float(np.quantile(values, 0.025)) if len(values) else math.nan,
                            "ci_high": float(np.quantile(values, 0.975)) if len(values) else math.nan,
                            "n_bootstrap": n_bootstrap,
                            "n_patients": int(labels_by_patient.shape[0]),
                            "n_images": int(len(frame)),
                            "n_positive_patients": int((labels_by_patient == 1).sum()),
                            "n_negative_patients": int((labels_by_patient == 0).sum()),
                            "seed": seed,
                        }
                    )
                reps.insert(0, "seed", seed)
                reps.insert(0, "threshold_name", threshold_name)
                reps.insert(0, "dataset", dataset)
                replicate_rows.append(reps)
    by_seed = pd.DataFrame(by_seed_rows)
    summary_rows = []
    for keys, group in by_seed.groupby(
        ["dataset", "evaluation_level", "grouping_policy", "threshold_policy", "threshold_name", "metric"],
        dropna=False,
    ):
        metric_reps = pd.concat(replicate_rows, ignore_index=True)
        dataset, _, _, _, threshold_name, metric = keys
        values = metric_reps[
            (metric_reps["dataset"] == dataset)
            & (metric_reps["threshold_name"] == threshold_name)
            & (metric_reps["metric"] == metric)
        ]["value"].dropna()
        first = group.iloc[0]
        summary_rows.append(
            {
                **dict(zip(
                    [
                        "dataset",
                        "evaluation_level",
                        "grouping_policy",
                        "threshold_policy",
                        "threshold_name",
                        "metric",
                    ],
                    keys,
                    strict=True,
                )),
                "mean": float(group["mean"].mean()),
                "bootstrap_mean": float(values.mean()) if len(values) else math.nan,
                "ci_low": float(values.quantile(0.025)) if len(values) else math.nan,
                "ci_high": float(values.quantile(0.975)) if len(values) else math.nan,
                "n_bootstrap": int(group["n_bootstrap"].sum()),
                "n_patients": int(first["n_patients"]),
                "n_images": int(first["n_images"]),
                "n_positive_patients": int(first["n_positive_patients"]),
                "n_negative_patients": int(first["n_negative_patients"]),
                "seed": "summary",
            }
        )
    write_table(by_seed, OUT / "patient_clustered_bootstrap_by_seed.csv")
    write_table(pd.DataFrame(summary_rows), OUT / "patient_clustered_bootstrap_summary.csv")
    write_table(pd.concat(replicate_rows, ignore_index=True), OUT / "patient_clustered_bootstrap_replicates.csv")


def patient_thresholds_from_calib(seed: int) -> pd.DataFrame:
    calib = load_predictions(seed, "cystods_calib", grouping="official_pid")
    rows = []
    image_thresholds = read_image_thresholds(seed)
    for aggregation_rule in AGGREGATIONS:
        base_image_threshold = image_thresholds["default_0.5"]
        scores = patient_score(calib, aggregation_rule, image_threshold=base_image_threshold)
        labels = scores["patient_true_label"].to_numpy(dtype=int)
        patient_scores = scores["patient_score"].to_numpy(dtype=float)
        for threshold_name in THRESHOLDS:
            threshold = select_threshold(labels, patient_scores, threshold_name)
            preds = (patient_scores >= threshold).astype(int)
            metrics = classification_metrics(labels, patient_scores, preds)
            rows.append(
                {
                    "seed": seed,
                    "aggregation_rule": aggregation_rule,
                    "threshold_policy": "patient_level_calibration_threshold",
                    "threshold_name": threshold_name,
                    "threshold_value": threshold,
                    "calibration_n_patients": int(len(scores)),
                    "calibration_n_positive_patients": int(labels.sum()),
                    "calibration_n_negative_patients": int((labels == 0).sum()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def evaluate_patient_aggregation() -> None:
    threshold_rows = []
    metric_rows = []
    prediction_rows = []
    for seed in SEEDS:
        patient_threshold_frame = patient_thresholds_from_calib(seed)
        threshold_rows.append(patient_threshold_frame)
        image_thresholds = read_image_thresholds(seed)
        for dataset in DATASETS:
            grouping = "case_level" if dataset == "ebtc_external" else "official_pid"
            frame = load_predictions(seed, dataset, grouping=grouping)
            for aggregation_rule in AGGREGATIONS:
                for threshold_policy, threshold_frame in [
                    (
                        "image_level_calibration_threshold",
                        pd.DataFrame(
                            {
                                "threshold_name": list(image_thresholds.keys()),
                                "threshold_value": list(image_thresholds.values()),
                            }
                        ),
                    ),
                    (
                        "patient_level_calibration_threshold",
                        patient_threshold_frame[
                            patient_threshold_frame["aggregation_rule"] == aggregation_rule
                        ][["threshold_name", "threshold_value"]],
                    ),
                ]:
                    base_image_threshold = image_thresholds["default_0.5"]
                    scores = patient_score(
                        frame,
                        aggregation_rule,
                        image_threshold=base_image_threshold,
                    )
                    labels = scores["patient_true_label"].to_numpy(dtype=int)
                    score_values = scores["patient_score"].to_numpy(dtype=float)
                    for _, threshold_row in threshold_frame.iterrows():
                        threshold_name = str(threshold_row["threshold_name"])
                        threshold_value = float(threshold_row["threshold_value"])
                        preds = (score_values >= threshold_value).astype(int)
                        metrics = classification_metrics(labels, score_values, preds)
                        metric_rows.append(
                            {
                                "dataset": dataset,
                                "split": dataset,
                                "grouping_policy": grouping,
                                "aggregation_rule": aggregation_rule,
                                "threshold_policy": threshold_policy,
                                "threshold_rule": threshold_name,
                                "threshold_value": threshold_value,
                                "n_patients": int(len(scores)),
                                "n_positive_patients": int(labels.sum()),
                                "n_negative_patients": int((labels == 0).sum()),
                                "seed": seed,
                                **metrics,
                            }
                        )
                        predictions = scores.copy()
                        predictions["dataset"] = dataset
                        predictions["grouping_policy"] = grouping
                        predictions["aggregation_rule"] = aggregation_rule
                        predictions["threshold_policy"] = threshold_policy
                        predictions["threshold_rule"] = threshold_name
                        predictions["threshold_value"] = threshold_value
                        predictions["patient_predicted_label"] = preds
                        predictions["seed"] = seed
                        prediction_rows.append(predictions)
    by_seed = pd.DataFrame(metric_rows)
    group_cols = [
        "dataset",
        "split",
        "grouping_policy",
        "aggregation_rule",
        "threshold_policy",
        "threshold_rule",
        "n_patients",
        "n_positive_patients",
        "n_negative_patients",
    ]
    metric_cols = [
        "threshold_value",
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
    summary = by_seed.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    write_table(pd.concat(threshold_rows, ignore_index=True), OUT / "patient_level_thresholds_from_calib.csv")
    write_table(by_seed, OUT / "patient_level_aggregation_by_seed.csv")
    write_table(summary, OUT / "patient_level_aggregation_summary.csv")
    write_table(pd.concat(prediction_rows, ignore_index=True), OUT / "patient_level_predictions_by_seed.csv")

    ensemble_predictions = []
    for dataset in DATASETS:
        grouping = "case_level" if dataset == "ebtc_external" else "official_pid"
        frame = load_ensemble(dataset, grouping=grouping)
        image_thresholds = {"default_0.5": 0.5}
        for name in ("sensitivity_90", "youden"):
            image_thresholds[name] = float(np.mean([read_image_thresholds(seed)[name] for seed in SEEDS]))
        for aggregation_rule in AGGREGATIONS:
            scores = patient_score(frame, aggregation_rule, image_threshold=image_thresholds["default_0.5"])
            for threshold_name, threshold_value in image_thresholds.items():
                preds = (scores["patient_score"].to_numpy(dtype=float) >= threshold_value).astype(int)
                rows = scores.copy()
                rows["dataset"] = dataset
                rows["grouping_policy"] = grouping
                rows["aggregation_rule"] = aggregation_rule
                rows["threshold_policy"] = "image_level_calibration_threshold"
                rows["threshold_rule"] = threshold_name
                rows["threshold_value"] = threshold_value
                rows["patient_predicted_label"] = preds
                rows["seed"] = "ensemble"
                ensemble_predictions.append(rows)
    write_table(pd.concat(ensemble_predictions, ignore_index=True), OUT / "patient_level_predictions_ensemble.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summarize_grouping()
    run_clustered_bootstrap(args.n_bootstrap)
    evaluate_patient_aggregation()
    note = (
        "# Corrected patient-level audit\n\n"
        "EBTC patient grouping was corrected for sensitivity analysis because the previous "
        "`pid` field collapsed legacy filenames without `pt_###` into one `unknown` group. "
        "The preferred EBTC grouping proxy in this audit is normalized `case_xxx`, which "
        "matches the dataset's TURBT case-level naming more closely than `pt` alone. "
        "A `case_pt` grouping is retained only as a sensitivity analysis because it expands "
        "the cohort into many lesion or part-level pseudo-groups.\n\n"
        "Patient-level aggregation is reported under two policies: image-level calibration "
        "thresholds reused on patient scores, and thresholds selected on CystoDS calibration "
        "patients for each aggregation rule. EBTC is never used for threshold selection.\n"
    )
    (OUT / "README_patient_level_corrected.md").write_text(note, encoding="utf-8")
    print(f"Wrote corrected patient-level audit to {OUT}")


if __name__ == "__main__":
    main()
