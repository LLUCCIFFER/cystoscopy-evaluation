"""Threshold sweep diagnostics for CDG-Cysto v2 validation predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdg_cysto.eval.thresholding import binary_confusion_at_threshold, threshold_grid  # noqa: E402
from cdg_cysto.utils.io import write_table  # noqa: E402


def _read_formal_leaderboard(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "run_type" in frame.columns:
        frame = frame[frame["run_type"].astype(str).eq("formal")].copy()
    if "run_dir" in frame.columns:
        frame = frame[~frame["run_dir"].astype(str).str.contains("smoke", case=False, na=False)].copy()
    if "run_id" in frame.columns:
        frame = frame[~frame["run_id"].astype(str).str.contains("smoke", case=False, na=False)].copy()
    return frame.reset_index(drop=True)


def _metadata(row: pd.Series) -> dict[str, Any]:
    return {
        "run_dir": row.get("run_dir", ""),
        "run_id": row.get("run_id", ""),
        "run_type": row.get("run_type", "formal"),
        "stage": row.get("stage", ""),
        "task": row.get("task", ""),
        "model_id": row.get("model_id", ""),
        "model_name": row.get("model_name", ""),
        "variant_id": row.get("variant_id", ""),
        "loss_name": row.get("loss_name", ""),
        "loss_tau": row.get("loss_tau", ""),
        "sampler_name": row.get("sampler_name", ""),
        "augmentation_name": row.get("augmentation_name", ""),
        "seed": row.get("seed", ""),
    }


def sweep_predictions(row: pd.Series) -> pd.DataFrame:
    predictions_path = Path(str(row["run_dir"])) / "predictions_val.csv"
    predictions = pd.read_csv(predictions_path)
    y_true = predictions["y_true"].to_numpy(dtype=int)
    y_score = predictions["prob_1"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    meta = _metadata(row)
    for threshold in threshold_grid(y_score):
        y_pred = (y_score >= float(threshold)).astype(int)
        metrics = binary_confusion_at_threshold(y_true, y_score, float(threshold))
        rows.append(
            {
                **meta,
                **metrics,
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "youden_index": float(metrics["sensitivity"] + metrics["specificity"] - 1.0),
            }
        )
    return pd.DataFrame(rows)


def target_operating_points(sweep: pd.DataFrame, targets: list[float]) -> pd.DataFrame:
    if sweep.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = ["run_dir", "model_name", "variant_id", "loss_name", "sampler_name", "seed"]
    for _, group in sweep.groupby(group_columns, dropna=False):
        for target in targets:
            feasible = group[group["sensitivity"] >= target].copy()
            if feasible.empty:
                best = group.sort_values(
                    ["sensitivity", "specificity", "threshold"],
                    ascending=[False, False, False],
                ).iloc[0]
                status = "target_not_reached"
            else:
                best = feasible.sort_values(
                    ["specificity", "threshold"],
                    ascending=[False, False],
                ).iloc[0]
                status = "target_reached"
            row = best.to_dict()
            row["target_sensitivity"] = float(target)
            row["target_status"] = status
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--targets-out", default="")
    parser.add_argument("--targets", default="0.85,0.90")
    args = parser.parse_args()

    leaderboard = _read_formal_leaderboard(args.leaderboard)
    frames = [sweep_predictions(row) for _, row in leaderboard.iterrows()]
    sweep = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_table(sweep, args.out)
    if args.targets_out:
        targets = [float(value) for value in args.targets.split(",") if value.strip()]
        write_table(target_operating_points(sweep, targets), args.targets_out)
    print(f"Wrote threshold sweep rows: {len(sweep)} -> {args.out}")


if __name__ == "__main__":
    main()
