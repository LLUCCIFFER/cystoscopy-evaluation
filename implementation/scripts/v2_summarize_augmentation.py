"""Summarize Stage 06 lightweight augmentation runs and by-modality metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdg_cysto.utils.io import write_table  # noqa: E402

SUMMARY_METRICS = [
    "val_auroc",
    "val_auprc",
    "val_macro_f1",
    "val_sensitivity_cancer",
    "val_specificity_cancer",
    "val_ece",
]

MODALITY_METRICS = [
    "balanced_accuracy",
    "macro_f1",
    "sensitivity_cancer",
    "specificity_cancer",
    "auroc",
    "auprc",
    "ece",
]


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    working = frame.copy()
    for column in columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    return working


def _mean_std_summary(frame: pd.DataFrame, group_columns: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=False))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group.columns else int(len(group))
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _load_by_modality(leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, run in leaderboard.iterrows():
        path = Path(str(run["run_dir"])) / "stage04_metrics_by_modality.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["run_id"] = run.get("run_id", Path(str(run["run_dir"])).name)
        frame["model_name"] = run.get("model_name", "")
        frame["variant_id"] = run.get("variant_id", "")
        frame["loss_name"] = run.get("loss_name", "")
        frame["sampler_name"] = run.get("sampler_name", "")
        frame["augmentation_name"] = run.get("augmentation_name", "")
        frame["seed"] = run.get("seed", "")
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


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


def _write_report(
    leaderboard_summary: pd.DataFrame,
    modality_summary: pd.DataFrame,
    output: Path,
) -> None:
    lines = [
        "# Stage 06 Augmentation Analysis",
        "",
        "Only formal Stage 06 runs are summarized. EBTC external validation is not used here.",
        "",
        "## Overall Validation Mean +/- Std",
        "",
    ]
    if leaderboard_summary.empty:
        lines.append("No Stage 06 leaderboard rows found.")
    else:
        columns = [
            "model_name",
            "variant_id",
            "augmentation_name",
            "n_seeds",
            "val_auroc_mean",
            "val_auprc_mean",
            "val_macro_f1_mean",
            "val_sensitivity_cancer_mean",
            "val_specificity_cancer_mean",
            "val_ece_mean",
        ]
        lines.append(_to_markdown_table(leaderboard_summary[[c for c in columns if c in leaderboard_summary.columns]]))
    lines.extend(["", "## By-Modality Mean +/- Std", ""])
    if modality_summary.empty:
        lines.append("No by-modality metrics found.")
    else:
        columns = [
            "model_name",
            "variant_id",
            "augmentation_name",
            "modality",
            "n_seeds",
            "sensitivity_cancer_mean",
            "specificity_cancer_mean",
            "macro_f1_mean",
            "auroc_mean",
            "auprc_mean",
        ]
        lines.append(_to_markdown_table(modality_summary[[c for c in columns if c in modality_summary.columns]]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard", default="outputs/v2/leaderboards/stage06_augmentation_leaderboard.csv")
    parser.add_argument("--out-dir", default="outputs/v2")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    leaderboard = pd.read_csv(args.leaderboard)
    if "run_type" in leaderboard.columns:
        leaderboard = leaderboard[leaderboard["run_type"].astype(str) == "formal"].copy()
    leaderboard = _numeric(leaderboard, SUMMARY_METRICS)
    group_columns = ["model_name", "variant_id", "augmentation_name"]
    leaderboard_summary = _mean_std_summary(leaderboard, group_columns, SUMMARY_METRICS)
    leaderboard_summary = leaderboard_summary.sort_values("val_auprc_mean", ascending=False)
    write_table(leaderboard_summary, out_dir / "leaderboards/stage06_augmentation_summary.csv")

    by_modality = _numeric(_load_by_modality(leaderboard), MODALITY_METRICS)
    write_table(by_modality, out_dir / "diagnostics/stage06_by_modality_metrics.csv")
    if not by_modality.empty:
        modality_summary = _mean_std_summary(
            by_modality,
            ["model_name", "variant_id", "augmentation_name", "modality"],
            MODALITY_METRICS,
        )
        modality_summary = modality_summary.sort_values(
            ["model_name", "variant_id", "augmentation_name", "modality"]
        )
    else:
        modality_summary = pd.DataFrame()
    write_table(modality_summary, out_dir / "diagnostics/stage06_by_modality_summary.csv")
    _write_report(
        leaderboard_summary,
        modality_summary,
        out_dir / "reports/stage06_augmentation_analysis.md",
    )
    print(f"Wrote Stage 06 augmentation summaries under {out_dir}")


if __name__ == "__main__":
    main()
