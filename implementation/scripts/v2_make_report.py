"""Create the final CDG-Cysto v2 experiment report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SECTIONS = [
    ("dataset audit", "outputs/v2/manifests/stage01_data_audit.md"),
    ("split audit", "outputs/v2/splits/stage02_split_audit.md"),
    ("backbone leaderboard", "outputs/v2/leaderboards/stage03_backbone_leaderboard.csv"),
    ("imbalance comparison", "outputs/v2/leaderboards/stage04_imbalance_leaderboard.csv"),
    ("augmentation comparison", "outputs/v2/leaderboards/stage04_augmentation_leaderboard.csv"),
    ("candidate selection", "outputs/v2/leaderboards/stage05_candidate_selection.md"),
    ("multi-seed stability", "outputs/v2/leaderboards/stage05_candidate_selection.csv"),
    ("calibration before/after", "outputs/v2/calibration/calibration_metrics_before_after.csv"),
    ("threshold analysis", "outputs/v2/calibration/threshold_metrics.csv"),
    ("CystoDS internal test", "outputs/v2/validation/cystods_internal/metrics_by_threshold.csv"),
    ("EBTC external validation", "outputs/v2/validation/ebtc_external/metrics_by_threshold.csv"),
    ("error analysis", "outputs/v2/error_analysis/error_summary.csv"),
    ("ROI binary", "outputs/v2/roi_binary/metrics_by_threshold.csv"),
    ("explainability", "outputs/v2/explainability/cam_metrics_summary.csv"),
    ("efficiency", "outputs/v2/efficiency/efficiency_metrics.csv"),
    ("statistical analysis", "outputs/v2/statistical/statistical_tests.csv"),
]

STATUS_ORDER = {
    "FAIL": 5,
    "BLOCKED": 4,
    "WARNING": 3,
    "PENDING": 2,
    "NOT_RUN": 1,
    "PASS": 0,
}


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "run_type" in frame.columns:
        frame = frame[frame["run_type"].astype(str) != "smoke"].copy()
    if "run_dir" in frame.columns:
        frame = frame[~frame["run_dir"].astype(str).str.contains("smoke", case=False, na=False)].copy()
    if "run_id" in frame.columns:
        frame = frame[~frame["run_id"].astype(str).str.contains("smoke", case=False, na=False)].copy()
    return frame


def _status(section: str, path: Path) -> str:
    if not path.exists():
        return "NOT_RUN" if section in {"multi-seed stability", "imbalance comparison", "augmentation comparison"} else "PENDING"
    if path.stat().st_size == 0:
        return "WARNING"
    if path.suffix.lower() != ".csv":
        if section == "candidate selection":
            csv_path = path.with_suffix(".csv")
            return _status(section, csv_path) if csv_path.exists() else "PENDING"
        return "PASS"
    frame = _read_csv(path)
    if frame.empty:
        return "NOT_RUN" if section in {"multi-seed stability", "imbalance comparison", "augmentation comparison"} else "PENDING"

    if section == "candidate selection":
        if "filter_status" in frame.columns and frame["filter_status"].eq("relaxed_no_candidate_met_all_filters").all():
            return "BLOCKED"
        return "PASS"
    if section == "multi-seed stability":
        if "model_name" not in frame.columns or "seed" not in frame.columns:
            return "NOT_RUN"
        seed_counts = frame.groupby("model_name")["seed"].nunique()
        if seed_counts.empty or seed_counts.lt(3).all():
            return "NOT_RUN"
        if seed_counts.lt(3).any():
            return "WARNING"
        return "PASS"
    if section == "imbalance comparison":
        loss_values = frame["loss_name"].nunique() if "loss_name" in frame.columns else 0
        sampler_values = frame["sampler_name"].nunique() if "sampler_name" in frame.columns else 0
        return "PASS" if max(loss_values, sampler_values) >= 2 else "NOT_RUN"
    if section == "augmentation comparison":
        aug_values = frame["augmentation_name"].nunique() if "augmentation_name" in frame.columns else 0
        return "PASS" if aug_values >= 2 else "NOT_RUN"
    return "PASS"


def _render_csv(path: Path, max_rows: int = 12) -> str:
    frame = _read_csv(path).head(max_rows)
    if path.exists() and frame.empty:
        return "_No formal rows._"
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)


def _render_artifact(path: Path) -> str:
    if not path.exists():
        return "_Pending._"
    if path.suffix.lower() == ".md":
        text = path.read_text(encoding="utf-8")
        return text[:4000]
    if path.suffix.lower() == ".csv":
        return _render_csv(path)
    return f"`{path}`"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-md", default="outputs/v2/reports/experiment_report_v2.md")
    args = parser.parse_args()

    rows = [(name, Path(path), _status(name, Path(path))) for name, path in SECTIONS]
    worst_status = max((row[2] for row in rows), key=lambda status: STATUS_ORDER[status])
    overall = "PASS" if worst_status == "PASS" else worst_status

    lines = [
        "# CDG-Cysto v2 Experiment Report",
        "",
        f"Overall conclusion: **{overall}**",
        "",
        "## Stage Status",
        "",
        "| section | status | artifact |",
        "| --- | --- | --- |",
    ]
    for name, path, status in rows:
        lines.append(f"| {name} | {status} | `{path}` |")
    lines.append("")
    for name, path, status in rows:
        lines.extend([f"## {name}", "", f"Status: **{status}**", "", _render_artifact(path), ""])

    output = Path(args.output_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote v2 report to {output}")


if __name__ == "__main__":
    main()
