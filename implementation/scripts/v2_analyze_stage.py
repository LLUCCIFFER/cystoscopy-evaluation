"""Analyze CDG-Cysto v2 run directories and emit diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdg_cysto.config.schema import load_yaml  # noqa: E402
from cdg_cysto.eval.metrics import compute_classification_metrics  # noqa: E402
from cdg_cysto.utils.io import read_json, read_table, write_json, write_table  # noqa: E402

STATUS_VALUES = ("PASS", "WARNING", "BLOCKED", "FAIL", "PENDING", "NOT_RUN")
REQUIRED_RUN_FILES = [
    "config_resolved.yaml",
    "run_manifest.json",
    "metrics_epoch.csv",
    "metrics_val.json",
    "predictions_val.csv",
    "logits_val.csv",
]
FIGURE_DIR = Path("outputs/v2/figures")


def _stage_key(stage: str) -> str:
    lowered = stage.lower()
    if "stage04" in lowered:
        return "stage04"
    if "stage03" in lowered:
        return "stage03"
    return lowered.replace("_", "-").split("-")[0] or "stage"


def _diagnostic_path(stage: str) -> Path:
    return Path(f"outputs/v2/diagnostics/{_stage_key(stage)}_prediction_distribution.csv")


def _naive_baseline_path(stage: str) -> Path:
    return Path(f"outputs/v2/diagnostics/{_stage_key(stage)}_naive_baselines.csv")


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except json.JSONDecodeError:
        return {}


def _safe_load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return load_yaml(path)
    except Exception:
        return {}


def _run_type(run_path: Path, config: dict[str, Any]) -> str:
    value = str(config.get("run_type", "")).lower()
    if value in {"smoke", "formal", "debug"}:
        return value
    return "smoke" if "smoke" in run_path.name.lower() else "formal"


def _is_smoke(run_path: Path, config: dict[str, Any], manifest: dict[str, Any] | None = None) -> bool:
    manifest = manifest or {}
    joined = " ".join(
        [
            run_path.name,
            str(config.get("run_type", "")),
            str(config.get("experiment_name", "")),
            str(manifest.get("run_id", "")),
            str(manifest.get("run_dir", "")),
        ]
    ).lower()
    return "smoke" in joined or str(config.get("run_type", "")).lower() == "smoke"


def _status_from_warnings(warnings: list[dict[str, str]]) -> str:
    if any(warning["level"] == "FAIL" for warning in warnings):
        return "FAIL"
    if any(warning["level"] == "BLOCKED" for warning in warnings):
        return "BLOCKED"
    if any(warning["level"] == "WARNING" for warning in warnings):
        return "WARNING"
    return "PASS"


def _specificity_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(y_true, dtype=int)
    predictions = np.asarray(y_pred, dtype=int)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return float(sensitivity), float(specificity)


def prediction_distribution(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    predictions = read_table(run_path / "predictions_val.csv")
    config = _safe_load_config(run_path / "config_resolved.yaml")
    manifest = _safe_read_json(run_path / "run_manifest.json")
    metrics = _safe_read_json(run_path / "metrics_val.json")
    y_true = predictions["y_true"].to_numpy(dtype=int)
    y_pred = predictions["y_pred"].to_numpy(dtype=int)
    prob_positive = (
        predictions["prob_1"].to_numpy(dtype=float)
        if "prob_1" in predictions.columns
        else np.full(len(predictions), np.nan)
    )
    sensitivity, specificity = _specificity_sensitivity(y_true, y_pred)
    try:
        auroc = float(roc_auc_score(y_true, prob_positive))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, prob_positive))
    except ValueError:
        auprc = float("nan")

    predicted_positive_rate = float((y_pred == 1).mean())
    collapse_flags: list[str] = []
    if predicted_positive_rate > 0.95:
        collapse_flags.append("COLLAPSED_TO_POSITIVE")
    if predicted_positive_rate < 0.05:
        collapse_flags.append("COLLAPSED_TO_NEGATIVE")
    if sensitivity == 1.0 and specificity < 0.1:
        collapse_flags.append("THRESHOLD_COLLAPSE_POSITIVE")
    if sensitivity < 0.2 and specificity > 0.9:
        collapse_flags.append("THRESHOLD_COLLAPSE_NEGATIVE")

    model = config.get("model", {}) or {}
    task = config.get("task", {}) or {}
    imbalance = config.get("imbalance", {}) or {}
    loss = imbalance.get("loss", {}) or {}
    sampler = imbalance.get("sampler", {}) or {}
    augmentation = config.get("augmentation", {}) or {}
    return {
        "run_dir": str(run_path),
        "run_id": manifest.get("run_id", run_path.name),
        "run_type": _run_type(run_path, config),
        "stage": config.get("stage", ""),
        "task_name": task.get("name", ""),
        "variant_id": config.get("variant_id", ""),
        "loss_name": loss.get("name", "cross_entropy"),
        "loss_tau": loss.get("tau", ""),
        "sampler_name": sampler.get("name", "none")
        if "name" in sampler
        else ("enabled" if sampler.get("enabled") else "none"),
        "augmentation_name": augmentation.get("name", augmentation.get("policy", "basic")),
        "model_id": model.get("model_id", ""),
        "model_name": model.get("name", run_path.name),
        "n": len(predictions),
        "predicted_positive_rate": predicted_positive_rate,
        "predicted_negative_rate": float((y_pred == 0).mean()),
        "mean_prob_positive": float(np.nanmean(prob_positive)),
        "std_prob_positive": float(np.nanstd(prob_positive)),
        "min_prob_positive": float(np.nanmin(prob_positive)),
        "max_prob_positive": float(np.nanmax(prob_positive)),
        "cancer_sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": auroc,
        "auprc": auprc,
        "val_auroc": metrics.get("auroc", auroc),
        "val_auprc": metrics.get("auprc", auprc),
        "val_macro_f1": metrics.get("macro_f1", float("nan")),
        "collapse_flags": ";".join(collapse_flags),
    }


def update_prediction_diagnostics(rows: list[dict[str, Any]], *, stage: str) -> pd.DataFrame:
    diagnostic_path = _diagnostic_path(stage)
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if diagnostic_path.exists():
        existing = pd.read_csv(diagnostic_path)
        if "run_dir" in existing.columns and not frame.empty:
            existing = existing[~existing["run_dir"].isin(frame["run_dir"])]
            frame = pd.concat([existing, frame], ignore_index=True)
    write_table(frame, diagnostic_path)
    return frame


def _safe_metric_dict(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    probabilities = np.column_stack([1.0 - y_score, y_score])
    metrics = compute_classification_metrics(
        y_true,
        y_pred=y_pred,
        y_score=probabilities,
        class_names=["non_cancer", "cancer"],
    )
    return {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "sensitivity": metrics["sensitivity_cancer"],
        "specificity": metrics["specificity_cancer"],
        "auroc": metrics.get("auroc", float("nan")),
        "auprc": metrics.get("auprc", float("nan")),
    }


def naive_baselines(predictions: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    prevalence = float((y_true == 1).mean())
    rng = np.random.default_rng(seed)
    rows = []
    baselines = {
        "all_negative": (np.zeros_like(y_true), np.zeros_like(y_true, dtype=float)),
        "all_positive": (np.ones_like(y_true), np.ones_like(y_true, dtype=float)),
        "prevalence_random": (
            (rng.random(len(y_true)) < prevalence).astype(int),
            np.full(len(y_true), prevalence, dtype=float),
        ),
    }
    for name, (y_pred, y_score) in baselines.items():
        row = {"baseline": name, "prevalence": prevalence}
        row.update(_safe_metric_dict(y_true, y_pred, y_score))
        rows.append(row)
    return pd.DataFrame(rows)


def write_naive_baselines(rows: list[pd.DataFrame], *, stage: str) -> pd.DataFrame:
    naive_baseline_path = _naive_baseline_path(stage)
    naive_baseline_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    dedup_columns = [
        column
        for column in ["stage", "task_name", "split", "baseline", "n_positive", "n_negative"]
        if column in frame.columns
    ]
    if dedup_columns:
        frame = frame.drop_duplicates(subset=dedup_columns, keep="first").reset_index(drop=True)
    write_table(frame, naive_baseline_path)
    return frame


def write_stage_confusion_matrices(rows: list[dict[str, Any]], *, stage: str) -> Path | None:
    if not rows:
        return None
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = FIGURE_DIR / f"{_stage_key(stage)}_confusion_matrices.png"
    columns = 3
    row_count = int(np.ceil(len(rows) / columns))
    fig, axes = plt.subplots(row_count, columns, figsize=(4 * columns, 4 * row_count), dpi=160)
    axes_array = np.atleast_1d(axes).reshape(-1)
    for axis in axes_array:
        axis.axis("off")
    for axis, row in zip(axes_array, rows, strict=False):
        predictions_path = Path(str(row["run_dir"])) / "predictions_val.csv"
        if not predictions_path.exists():
            continue
        predictions = pd.read_csv(predictions_path)
        matrix = confusion_matrix(
            predictions["y_true"].to_numpy(dtype=int),
            predictions["y_pred"].to_numpy(dtype=int),
            labels=[0, 1],
        )
        axis.axis("on")
        ConfusionMatrixDisplay(matrix, display_labels=["non_cancer", "cancer"]).plot(
            ax=axis,
            colorbar=False,
        )
        title_bits = [str(row.get("model_name", ""))]
        if row.get("variant_id"):
            title_bits.append(str(row["variant_id"]))
        axis.set_title("\n".join(title_bits), fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    return figure_path


def _frame_markdown(frame: pd.DataFrame, *, max_rows: int = 12) -> str:
    if frame.empty:
        return "_No rows._"
    subset = frame.head(max_rows)
    columns = [str(column) for column in subset.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in subset.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in subset.columns) + " |")
    return "\n".join(lines)


def analyze_run(run_dir: str | Path) -> tuple[str, dict[str, Any]]:
    run_path = Path(run_dir)
    warnings: list[dict[str, str]] = []
    lines = [f"# Stage Analysis: {run_path.name}", ""]
    config = _safe_load_config(run_path / "config_resolved.yaml")
    manifest = _safe_read_json(run_path / "run_manifest.json")
    if _is_smoke(run_path, config, manifest):
        warnings.append({"level": "WARNING", "message": "Smoke run detected; exclude from formal leaderboard."})

    for filename in REQUIRED_RUN_FILES:
        path = run_path / filename
        status = "present" if path.exists() and path.stat().st_size > 0 else "missing_or_empty"
        lines.append(f"- `{filename}`: {status}")
        if status != "present":
            warnings.append({"level": "FAIL", "message": f"Missing or empty required file: {filename}"})

    required_manifest_keys = ["run_id", "config_hash", "split_hash", "git_commit", "seed", "command"]
    for key in required_manifest_keys:
        if key not in manifest:
            warnings.append({"level": "FAIL", "message": f"run_manifest.json missing key: {key}"})
    if manifest:
        lines.extend(["", "## Manifest", ""])
        for key in required_manifest_keys:
            lines.append(f"- `{key}`: {manifest.get(key, 'MISSING')}")

    metrics = _safe_read_json(run_path / "metrics_val.json")
    if metrics:
        lines.extend(["", "## Validation Metrics", ""])
        for key, value in sorted(metrics.items()):
            lines.append(f"- `{key}`: {value}")

    predictions_path = run_path / "predictions_val.csv"
    if predictions_path.exists() and predictions_path.stat().st_size > 0:
        try:
            predictions = read_table(predictions_path)
            lines.extend(["", "## Predictions", ""])
            lines.append(f"- rows: {len(predictions)}")
            if {"y_true", "y_pred"}.issubset(predictions.columns):
                accuracy = float((predictions["y_true"] == predictions["y_pred"]).mean())
                lines.append(f"- quick_accuracy: {accuracy:.6f}")
            diagnostic = prediction_distribution(run_path)
            update_prediction_diagnostics([diagnostic], stage=str(diagnostic.get("stage") or "run"))
            lines.extend(["", "## Collapse Diagnostics", ""])
            for key, value in diagnostic.items():
                if key in {"run_dir", "run_id"}:
                    continue
                lines.append(f"- `{key}`: {value}")
            if diagnostic["collapse_flags"]:
                warnings.append(
                    {
                        "level": "WARNING",
                        "message": f"Prediction collapse flags: {diagnostic['collapse_flags']}",
                    }
                )
        except pd.errors.EmptyDataError:
            warnings.append({"level": "FAIL", "message": "predictions_val.csv has no columns"})

    status = _status_from_warnings(warnings)
    warning_payload = {
        "run_dir": str(run_path),
        "status": status,
        "warnings": warnings,
    }
    return "\n".join(lines) + "\n", warning_payload


def _stage_matches(run_path: Path, config: dict[str, Any], stage: str) -> bool:
    stage_text = str(config.get("stage", "")).lower()
    return stage.lower() in stage_text or stage.lower() in run_path.name.lower()


def _formal_stage_runs(run_root: Path, stage: str) -> list[Path]:
    runs: list[Path] = []
    for run_path in sorted(path for path in run_root.iterdir() if path.is_dir()):
        config = _safe_load_config(run_path / "config_resolved.yaml")
        manifest = _safe_read_json(run_path / "run_manifest.json")
        if _is_smoke(run_path, config, manifest):
            continue
        if _run_type(run_path, config) != "formal":
            continue
        if _stage_matches(run_path, config, stage):
            runs.append(run_path)
    return runs


def _leaderboard_smoke_warning() -> dict[str, str] | None:
    leaderboard = Path("outputs/v2/leaderboards/stage03_backbone_leaderboard.csv")
    if not leaderboard.exists():
        return None
    try:
        frame = pd.read_csv(leaderboard)
    except pd.errors.EmptyDataError:
        return None
    has_smoke = False
    if "run_type" in frame.columns:
        has_smoke = has_smoke or frame["run_type"].astype(str).eq("smoke").any()
    if "run_dir" in frame.columns:
        has_smoke = has_smoke or frame["run_dir"].astype(str).str.contains("smoke", case=False, na=False).any()
    if has_smoke:
        return {"level": "WARNING", "message": "Formal leaderboard contains smoke rows."}
    return None


def _formal_passes(diagnostic: dict[str, Any]) -> bool:
    flags = str(diagnostic.get("collapse_flags", ""))
    return bool(
        float(diagnostic.get("val_auroc", 0.0)) >= 0.85
        and float(diagnostic.get("val_auprc", 0.0)) >= 0.35
        and float(diagnostic.get("val_macro_f1", 0.0)) >= 0.55
        and 0.05 <= float(diagnostic.get("predicted_positive_rate", 0.0)) <= 0.50
        and not flags
    )


def analyze_stage(run_root: str | Path, stage: str, output_path: str | Path) -> dict[str, Any]:
    run_root_path = Path(run_root)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[dict[str, str]] = []
    smoke_warning = _leaderboard_smoke_warning()
    if smoke_warning is not None:
        warnings.append(smoke_warning)

    runs = _formal_stage_runs(run_root_path, stage)
    if not runs:
        payload = {"stage": stage, "status": "PENDING", "warnings": warnings}
        out.write_text(f"# {stage} Analysis\n\nStatus: **PENDING**\n\nNo formal runs found.\n", encoding="utf-8")
        write_json(payload, out.with_suffix(".warnings.json"))
        return payload

    diagnostics: list[dict[str, Any]] = []
    baseline_frames: list[pd.DataFrame] = []
    lines = [
        f"# {stage} Formal Analysis",
        "",
        "Accuracy is not a primary metric in this imbalanced task; balanced accuracy, AUROC, AUPRC, Macro-F1, sensitivity, specificity, and collapse diagnostics are prioritized.",
        "",
        "## Runs",
        "",
        "| model | variant | loss | sampler | AUROC | AUPRC | Macro-F1 | positive_rate | sensitivity | specificity | collapse_flags |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for run_path in runs:
        markdown, run_payload = analyze_run(run_path)
        (run_path / "stage_analysis.md").write_text(markdown, encoding="utf-8")
        write_json(run_payload, run_path / "stage_warnings.json")
        warnings.extend(run_payload["warnings"])
        diagnostic = prediction_distribution(run_path)
        diagnostics.append(diagnostic)
        predictions = read_table(run_path / "predictions_val.csv")
        baselines = naive_baselines(predictions)
        y_true = predictions["y_true"].to_numpy(dtype=int)
        baselines.insert(0, "n_negative", int((y_true == 0).sum()))
        baselines.insert(0, "n_positive", int((y_true == 1).sum()))
        baselines.insert(0, "split", "val")
        baselines.insert(0, "task_name", diagnostic["task_name"])
        baselines.insert(0, "stage", diagnostic["stage"])
        baseline_frames.append(baselines)
        lines.append(
            f"| {diagnostic['model_name']} | {diagnostic['variant_id'] or '-'} | "
            f"{diagnostic['loss_name']} | {diagnostic['sampler_name']} | "
            f"{float(diagnostic['val_auroc']):.6f} | "
            f"{float(diagnostic['val_auprc']):.6f} | {float(diagnostic['val_macro_f1']):.6f} | "
            f"{float(diagnostic['predicted_positive_rate']):.6f} | "
            f"{float(diagnostic['cancer_sensitivity']):.6f} | {float(diagnostic['specificity']):.6f} | "
            f"{diagnostic['collapse_flags'] or 'none'} |"
        )

    diagnostic_frame = update_prediction_diagnostics(diagnostics, stage=stage)
    naive_frame = write_naive_baselines(baseline_frames, stage=stage)
    confusion_path = write_stage_confusion_matrices(diagnostics, stage=stage)
    passed = [diagnostic for diagnostic in diagnostics if _formal_passes(diagnostic)]
    if not passed:
        warnings.append(
            {
                "level": "BLOCKED",
                "message": "No formal Stage 03 model met minimum progression criteria.",
            }
        )
    status = _status_from_warnings(warnings)

    lines.extend(
        [
            "",
            "## Naive Baselines",
            "",
            "All-negative can have high accuracy under class imbalance; do not use accuracy as the primary metric.",
            "",
            _frame_markdown(naive_frame),
            "",
            "## Diagnostics CSV",
            "",
            f"- `{_diagnostic_path(stage)}`",
            f"- `{_naive_baseline_path(stage)}`",
            f"- `{confusion_path}`" if confusion_path is not None else "- confusion matrix figure: not generated",
            "",
            f"Stage status: **{status}**",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "stage": stage,
        "status": status,
        "runs": [str(path) for path in runs],
        "n_runs": len(runs),
        "n_passed_progression": len(passed),
        "diagnostics_path": str(_diagnostic_path(stage)),
        "naive_baseline_path": str(_naive_baseline_path(stage)),
        "confusion_matrix_path": str(confusion_path) if confusion_path is not None else "",
        "warnings": warnings,
    }
    write_json(payload, out.with_suffix(".warnings.json"))
    write_table(diagnostic_frame, _diagnostic_path(stage))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    parser.add_argument("--run-root")
    parser.add_argument("--stage", default="stage03")
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.run_root:
        output = args.out or f"outputs/v2/reports/{args.stage}_analysis.md"
        payload = analyze_stage(args.run_root, args.stage, output)
        print(f"Wrote {output}")
        print(f"status={payload['status']}")
        return

    if not args.run_dir:
        parser.error("Either --run-dir or --run-root is required.")
    markdown, warnings = analyze_run(args.run_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stage_analysis.md").write_text(markdown, encoding="utf-8")
    write_json(warnings, run_dir / "stage_warnings.json")
    print(f"Wrote {run_dir / 'stage_analysis.md'}")
    print(f"Wrote {run_dir / 'stage_warnings.json'}")


if __name__ == "__main__":
    main()
