"""Select top CDG-Cysto v2 candidate backbones from leaderboards."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

STATUS_VALUES = {"PASS", "WARNING", "BLOCKED", "FAIL", "PENDING", "NOT_RUN"}


def selection_score(row: pd.Series) -> float:
    return float(
        0.25 * row.get("val_auroc", 0.0)
        + 0.20 * row.get("val_auprc", 0.0)
        + 0.20 * row.get("val_macro_f1", 0.0)
        + 0.20 * row.get("val_sensitivity_cancer", row.get("val_sensitivity_macro", 0.0))
        + 0.10 * row.get("val_specificity_cancer", row.get("val_specificity_macro", 0.0))
        + 0.05 * (1.0 - row.get("val_ece", 1.0))
    )


def load_leaderboards(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if Path(path).exists()]
    if not frames:
        msg = f"No leaderboard files found: {paths}"
        raise FileNotFoundError(msg)
    frame = pd.concat(frames, ignore_index=True)
    if "run_type" in frame.columns:
        frame = frame[frame["run_type"].astype(str) == "formal"].copy()
    if "run_dir" in frame.columns:
        frame = frame[~frame["run_dir"].astype(str).str.contains("smoke", case=False, na=False)].copy()
    if "run_id" in frame.columns:
        frame = frame[~frame["run_id"].astype(str).str.contains("smoke", case=False, na=False)].copy()
    if frame.empty:
        msg = "No formal leaderboard rows available for candidate selection."
        raise ValueError(msg)
    frame = frame.drop_duplicates(subset=["run_dir", "seed", "model_id", "model_name"])
    frame["selection_score"] = frame.apply(selection_score, axis=1)
    return frame


def filter_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        (frame["val_auroc"] >= 0.85)
        & (frame["val_auprc"] >= 0.35)
        & (frame["val_macro_f1"] >= 0.55)
        & (frame["val_ece"] <= 0.15)
    ].copy()


def _candidate_yaml(row: pd.Series) -> dict[str, Any]:
    return {
        "selected_candidate": {
            "rank": int(row["rank"]),
            "run_dir": row["run_dir"],
            "model_id": row["model_id"],
            "model_name": row["model_name"],
            "variant_id": row.get("variant_id", ""),
            "loss_name": row.get("loss_name", ""),
            "loss_tau": row.get("loss_tau", ""),
            "sampler_name": row.get("sampler_name", ""),
            "run_type": row.get("run_type", "formal"),
            "filter_status": row.get("filter_status", ""),
            "selection_score": float(row["selection_score"]),
            "val_auroc": float(row["val_auroc"]),
            "val_auprc": float(row["val_auprc"]),
            "val_macro_f1": float(row["val_macro_f1"]),
            "val_sensitivity_cancer": float(row["val_sensitivity_cancer"]),
            "val_specificity_cancer": float(row["val_specificity_cancer"]),
            "val_ece": float(row["val_ece"]),
        }
    }


def write_selection_outputs(frame: pd.DataFrame, output_dir: str | Path, *, top_k: int = 4) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates = filter_candidates(frame)
    if candidates.empty:
        candidates = frame.copy()
        candidates["filter_status"] = "relaxed_no_candidate_met_all_filters"
        stage_status = "BLOCKED"
    else:
        candidates["filter_status"] = "passed"
        stage_status = "PASS"
    candidates = candidates.sort_values("selection_score", ascending=False).head(top_k).reset_index(drop=True)
    candidates["rank"] = candidates.index + 1
    candidates["stage_status"] = stage_status
    for _, row in candidates.iterrows():
        path = output / f"stage05_candidate_top{int(row['rank'])}.yaml"
        path.write_text(yaml.safe_dump(_candidate_yaml(row), sort_keys=False), encoding="utf-8")
    lines = ["# Stage 05 Candidate Selection", "", f"Stage status: **{stage_status}**", ""]
    lines.append(
        "Formula: 0.25*AUROC + 0.20*AUPRC + 0.20*MacroF1 + 0.20*CancerSensitivity + "
        "0.10*Specificity + 0.05*(1-ECE)."
    )
    lines.append("")
    lines.append("| rank | model | variant | selection_score | filter_status | run_dir |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for _, row in candidates.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['model_name']} | {row.get('variant_id', '')} | "
            f"{row['selection_score']:.6f} | "
            f"{row['filter_status']} | {row['run_dir']} |"
        )
    (output / "stage05_candidate_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    candidates.to_csv(output / "stage05_candidate_selection.csv", index=False)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leaderboards",
        nargs="*",
        default=[
            "outputs/v2/leaderboards/stage04_imbalance_leaderboard.csv",
        ],
    )
    parser.add_argument("--output-dir", default="outputs/v2/leaderboards")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    frame = load_leaderboards(args.leaderboards)
    candidates = write_selection_outputs(frame, args.output_dir, top_k=args.top_k)
    print(
        candidates[["rank", "model_name", "variant_id", "selection_score", "filter_status"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
