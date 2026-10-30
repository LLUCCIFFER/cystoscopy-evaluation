"""Summarize CDG-Cysto v2 multi-seed candidate verification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cdg_cysto.eval.calibration import calibration_metrics  # noqa: E402
from cdg_cysto.eval.metrics import (  # noqa: E402
    save_confusion_matrix_plot,
    save_pr_curve_plot,
    save_reliability_diagram,
    save_roc_curve_plot,
)
from cdg_cysto.utils.io import write_table  # noqa: E402
from scripts.v2_analyze_stage import prediction_distribution  # noqa: E402

SUMMARY_METRICS = [
    "val_auroc",
    "val_auprc",
    "val_macro_f1",
    "val_sensitivity_cancer",
    "val_specificity_cancer",
    "val_ece",
    "val_selection_score",
    "selection_score_v2",
]


def _slug(value: str) -> str:
    return "_".join(part for part in "".join(c if c.isalnum() else "_" for c in value).split("_") if part)


def _read_formal(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if Path(path).exists()]
    if not frames:
        msg = f"No leaderboard files found: {paths}"
        raise FileNotFoundError(msg)
    frame = pd.concat(frames, ignore_index=True)
    if "run_type" in frame.columns:
        frame = frame[frame["run_type"].astype(str).eq("formal")].copy()
    if "run_dir" in frame.columns:
        frame = frame[~frame["run_dir"].astype(str).str.contains("smoke", case=False, na=False)].copy()
    if "run_id" in frame.columns:
        frame = frame[~frame["run_id"].astype(str).str.contains("smoke", case=False, na=False)].copy()
    if frame.empty:
        msg = "No formal rows remain after smoke filtering."
        raise ValueError(msg)
    return frame.reset_index(drop=True)


def _selection_score_v2(row: pd.Series) -> float:
    return float(
        0.25 * row.get("val_auroc", 0.0)
        + 0.20 * row.get("val_auprc", 0.0)
        + 0.20 * row.get("val_macro_f1", 0.0)
        + 0.20 * row.get("val_sensitivity_cancer", row.get("val_sensitivity_macro", 0.0))
        + 0.10 * row.get("val_specificity_cancer", row.get("val_specificity_macro", 0.0))
        + 0.05 * (1.0 - row.get("val_ece", 1.0))
    )


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith("prob_")],
        key=lambda value: int(value.split("_")[1]),
    )


def _logit_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith("logit_")],
        key=lambda value: int(value.split("_")[1]),
    )


def _write_seed_figures(row: pd.Series, figures_dir: Path) -> dict[str, str]:
    run_dir = Path(str(row["run_dir"]))
    predictions = pd.read_csv(run_dir / "predictions_val.csv")
    prob_columns = _probability_columns(predictions)
    probabilities = predictions[prob_columns].to_numpy(dtype=float)
    labels = predictions["y_true"].to_numpy(dtype=int)
    predictions_argmax = predictions["y_pred"].to_numpy(dtype=int)
    slug = _slug(f"{row['model_name']}_{row.get('variant_id', '')}_seed{row['seed']}")
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "confusion_matrix": str(figures_dir / f"{slug}_confusion_matrix.png"),
        "roc_curve": str(figures_dir / f"{slug}_roc_curve.png"),
        "pr_curve": str(figures_dir / f"{slug}_pr_curve.png"),
        "reliability": str(figures_dir / f"{slug}_reliability.png"),
    }
    save_confusion_matrix_plot(labels, predictions_argmax, ["non_cancer", "cancer"], paths["confusion_matrix"])
    save_roc_curve_plot(labels, probabilities, paths["roc_curve"])
    save_pr_curve_plot(labels, probabilities, paths["pr_curve"])
    save_reliability_diagram(labels, probabilities, paths["reliability"])
    return paths


def _seed_calibration_row(row: pd.Series) -> dict[str, Any]:
    run_dir = Path(str(row["run_dir"]))
    logits_frame = pd.read_csv(run_dir / "logits_val.csv")
    logits = logits_frame[_logit_columns(logits_frame)].to_numpy(dtype=float)
    labels = logits_frame["y_true"].to_numpy(dtype=int)
    metrics = calibration_metrics(labels, logits)
    return {
        "run_dir": str(run_dir),
        "run_id": row.get("run_id", ""),
        "model_name": row.get("model_name", ""),
        "variant_id": row.get("variant_id", ""),
        "seed": int(row.get("seed", 0)),
        "ece": metrics["ece"],
        "brier": metrics["brier"],
        "nll": metrics["nll"],
    }


def _mean_std_text(mean: float, std: float) -> str:
    if np.isnan(std):
        return f"{mean:.6f} +/- NA"
    return f"{mean:.6f} +/- {std:.6f}"


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    for column in [
        "val_auroc",
        "val_auprc",
        "val_macro_f1",
        "val_sensitivity_cancer",
        "val_specificity_cancer",
        "val_ece",
        "val_selection_score",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["selection_score_v2"] = frame.apply(_selection_score_v2, axis=1)
    rows: list[dict[str, Any]] = []
    group_columns = ["model_name", "variant_id", "loss_name", "loss_tau", "sampler_name", "augmentation_name"]
    for keys, group in frame.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys, strict=False))
        seeds = sorted(int(seed) for seed in group["seed"].astype(int).unique())
        row["n_seeds"] = len(seeds)
        row["seeds"] = ",".join(str(seed) for seed in seeds)
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values.dropna()) > 1 else float("nan")
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_mean_std"] = _mean_std_text(mean, std)
        auroc_std = float(row["val_auroc_std"])
        macro_std = float(row["val_macro_f1_std"])
        row["unstable"] = bool(
            row["n_seeds"] < 3
            or (np.isfinite(auroc_std) and auroc_std > 0.02)
            or (np.isfinite(macro_std) and macro_std > 0.03)
        )
        row["stability_status"] = "UNSTABLE" if row["unstable"] else "STABLE"
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values("selection_score_v2_mean", ascending=False).reset_index(drop=True)


def _write_report(summary: pd.DataFrame, report_path: Path) -> None:
    lines = [
        "# Stage 05 Multi-Seed Verification",
        "",
        "Unstable rule: AUROC std > 0.02 or Macro-F1 std > 0.03, or fewer than 3 seeds.",
        "",
        "| rank | model | variant | seeds | AUROC | AUPRC | Macro-F1 | Sens | Spec | ECE | stability |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, row in summary.iterrows():
        lines.append(
            f"| {index + 1} | {row['model_name']} | {row['variant_id']} | {row['seeds']} | "
            f"{row['val_auroc_mean_std']} | {row['val_auprc_mean_std']} | "
            f"{row['val_macro_f1_mean_std']} | {row['val_sensitivity_cancer_mean_std']} | "
            f"{row['val_specificity_cancer_mean_std']} | {row['val_ece_mean_std']} | "
            f"{row['stability_status']} |"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboards", nargs="+", required=True)
    parser.add_argument("--summary-out", default="outputs/v2/leaderboards/stage05_multiseed_summary.csv")
    parser.add_argument("--diagnostics-out", default="outputs/v2/diagnostics/stage05_prediction_distribution.csv")
    parser.add_argument(
        "--calibration-out",
        default="outputs/v2/diagnostics/stage05_seed_calibration_metrics.csv",
    )
    parser.add_argument("--stable-top2-out", default="outputs/v2/leaderboards/stage05_stable_top2.csv")
    parser.add_argument("--report-out", default="outputs/v2/reports/stage05_multiseed_analysis.md")
    parser.add_argument("--figures-dir", default="outputs/v2/figures/stage05_multiseed")
    args = parser.parse_args()

    frame = _read_formal(args.leaderboards)
    summary = summarize(frame)
    write_table(summary, args.summary_out)
    stable_top2 = summary[summary["stability_status"].eq("STABLE")].head(2).copy()
    write_table(stable_top2, args.stable_top2_out)

    diagnostics: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    figures_dir = Path(args.figures_dir)
    for _, row in frame.iterrows():
        diagnostics.append(prediction_distribution(row["run_dir"]))
        calibration_row = _seed_calibration_row(row)
        calibration_row.update(_write_seed_figures(row, figures_dir))
        calibration_rows.append(calibration_row)
    write_table(pd.DataFrame(diagnostics), args.diagnostics_out)
    write_table(pd.DataFrame(calibration_rows), args.calibration_out)
    _write_report(summary, Path(args.report_out))
    print(summary[["model_name", "variant_id", "selection_score_v2_mean", "stability_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
