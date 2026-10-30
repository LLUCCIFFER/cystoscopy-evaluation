"""Error analysis helpers for CDG-Cysto v2 predictions."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def merge_predictions_metadata(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(metadata, on="sample_id", how="left", suffixes=("", "_meta"))
    prob_columns = [column for column in merged.columns if column.startswith("prob_")]
    merged["confidence"] = merged[prob_columns].max(axis=1) if prob_columns else np.nan
    if "prob_1" in merged.columns:
        merged["cancer_score"] = merged["prob_1"]
    else:
        merged["cancer_score"] = merged["confidence"]
    merged["is_correct"] = merged["y_true"].astype(int) == merged["y_pred"].astype(int)
    merged["is_false_negative"] = (merged["y_true"].astype(int) == 1) & (merged["y_pred"].astype(int) == 0)
    merged["is_false_positive"] = (merged["y_true"].astype(int) == 0) & (merged["y_pred"].astype(int) == 1)
    return merged


def summarize_errors(merged: pd.DataFrame) -> pd.DataFrame:
    total = len(merged)
    errors = int((~merged["is_correct"]).sum())
    rows = [
        {
            "n": total,
            "errors": errors,
            "error_rate": errors / total if total else np.nan,
            "false_negatives": int(merged["is_false_negative"].sum()),
            "false_positives": int(merged["is_false_positive"].sum()),
        }
    ]
    return pd.DataFrame(rows)


def error_by_group(merged: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for value, group in merged.groupby(group_column, dropna=False):
        errors = int((~group["is_correct"]).sum())
        rows.append(
            {
                group_column: value,
                "n": len(group),
                "errors": errors,
                "error_rate": errors / len(group) if len(group) else np.nan,
                "false_negatives": int(group["is_false_negative"].sum()),
                "false_positives": int(group["is_false_positive"].sum()),
            }
        )
    return pd.DataFrame(rows)


def select_error_cases(merged: pd.DataFrame, *, top_k: int = 20) -> dict[str, pd.DataFrame]:
    high_confidence_errors = (
        merged[~merged["is_correct"]].sort_values("confidence", ascending=False).head(top_k)
    )
    low_confidence_correct = (
        merged[merged["is_correct"]].sort_values("confidence", ascending=True).head(top_k)
    )
    top_false_negatives = (
        merged[merged["is_false_negative"]].sort_values("cancer_score", ascending=True).head(top_k)
    )
    top_false_positives = (
        merged[merged["is_false_positive"]].sort_values("cancer_score", ascending=False).head(top_k)
    )
    return {
        "high_confidence_errors": high_confidence_errors,
        "low_confidence_correct": low_confidence_correct,
        "top_false_negatives": top_false_negatives,
        "top_false_positives": top_false_positives,
    }


def copy_case_images(frame: pd.DataFrame, output_dir: str | Path, *, prefix: str) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for rank, (_, row) in enumerate(frame.iterrows(), start=1):
        source = Path(str(row["image_path"]))
        if not source.exists():
            continue
        destination = target / f"{rank:03d}_{prefix}_{row['sample_id']}{source.suffix.lower()}"
        shutil.copy2(source, destination)
