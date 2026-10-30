"""Run final CDG-Cysto v2 calibration, internal test, and EBTC external validation."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdg_cysto.utils.io import write_table  # noqa: E402


METRIC_COLUMNS = [
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
]


def _run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


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


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _mean_std(frame: pd.DataFrame, group_columns: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
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


def _select_runs(args: argparse.Namespace) -> pd.DataFrame:
    leaderboard = pd.read_csv(args.leaderboard)
    if "run_type" in leaderboard.columns:
        leaderboard = leaderboard[leaderboard["run_type"].astype(str) == "formal"].copy()
    selected = leaderboard[
        (leaderboard["model_name"].astype(str) == args.model_name)
        & (leaderboard["variant_id"].astype(str) == args.variant_id)
        & (leaderboard["augmentation_name"].astype(str) == args.augmentation_name)
    ].copy()
    selected["seed"] = selected["seed"].astype(int)
    selected = selected[selected["seed"].isin(args.seeds)].sort_values("seed")
    missing = sorted(set(args.seeds) - set(selected["seed"].tolist()))
    if missing:
        msg = f"Missing selected runs for seeds: {missing}"
        raise RuntimeError(msg)
    return selected


def _run_one(row: pd.Series, args: argparse.Namespace, out_dir: Path) -> None:
    run_id = str(row["run_id"])
    seed = int(row["seed"])
    run_dir = Path(str(row["run_dir"]))
    checkpoint = run_dir / "checkpoint_best.pt"
    if not checkpoint.exists():
        msg = f"Missing checkpoint: {checkpoint}"
        raise FileNotFoundError(msg)
    seed_dir = out_dir / f"seed_{seed}"
    calib_raw = seed_dir / "calib_raw"
    calibration = seed_dir / "calibration"
    internal = seed_dir / "cystods_internal"
    external = seed_dir / "ebtc_external"
    _run(
        [
            sys.executable,
            "scripts/v2_collect_logits.py",
            "--checkpoint",
            str(checkpoint),
            "--split-csv",
            args.calib_split,
            "--output-dir",
            str(calib_raw),
            "--label-column",
            args.label_column,
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/v2_calibrate.py",
            "--calib-logits",
            str(calib_raw / "logits.csv"),
            "--output-dir",
            str(calibration),
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/v2_select_threshold.py",
            "--calib-predictions",
            str(calibration / "predictions_calibrated.csv"),
            "--output-dir",
            str(calibration),
        ]
    )
    common_validate = [
        "--checkpoint",
        str(checkpoint),
        "--temperature-json",
        str(calibration / "temperature.json"),
        "--thresholds-json",
        str(calibration / "thresholds.json"),
        "--label-column",
        args.label_column,
    ]
    _run(
        [
            sys.executable,
            "scripts/v2_external_validate.py",
            *common_validate,
            "--split-csv",
            args.internal_split,
            "--output-dir",
            str(internal),
        ]
    )
    _run(
        [
            sys.executable,
            "scripts/v2_external_validate.py",
            *common_validate,
            "--split-csv",
            args.external_split,
            "--output-dir",
            str(external),
            "--external-ebtc",
        ]
    )
    (seed_dir / "source_run_id.txt").write_text(run_id + "\n", encoding="utf-8")


def _collect_outputs(selected: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold_rows: list[pd.DataFrame] = []
    validation_rows: list[pd.DataFrame] = []
    ebtc_rows: list[pd.DataFrame] = []
    for _, row in selected.iterrows():
        seed = int(row["seed"])
        seed_dir = out_dir / f"seed_{seed}"
        threshold = _read_optional(seed_dir / "calibration" / "threshold_metrics.csv")
        if not threshold.empty:
            threshold["seed"] = seed
            threshold_rows.append(threshold)
        for dataset_name, subdir in [
            ("cystods_internal", "cystods_internal"),
            ("ebtc_external", "ebtc_external"),
        ]:
            metrics = _read_optional(seed_dir / subdir / "metrics_by_threshold.csv")
            if not metrics.empty:
                metrics["seed"] = seed
                metrics["dataset"] = dataset_name
                validation_rows.append(metrics)
        ebtc = _read_optional(seed_dir / "ebtc_external" / "metrics_by_ebtc_class.csv")
        if not ebtc.empty:
            ebtc["seed"] = seed
            ebtc_rows.append(ebtc)
    thresholds = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame()
    validation = pd.concat(validation_rows, ignore_index=True) if validation_rows else pd.DataFrame()
    ebtc_by_class = pd.concat(ebtc_rows, ignore_index=True) if ebtc_rows else pd.DataFrame()
    return thresholds, validation, ebtc_by_class


def _write_report(
    selected: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    validation_summary: pd.DataFrame,
    ebtc_class_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    method = selected.iloc[0]
    lines = [
        "# CDG-Cysto v2 Final Validation",
        "",
        "Final method was selected from CystoDS validation only. EBTC was not used for model selection, calibration, or threshold selection.",
        "",
        "## Selected Method",
        "",
        f"- model_name: {method['model_name']}",
        f"- variant_id: {method['variant_id']}",
        f"- augmentation_name: {method['augmentation_name']}",
        f"- seeds: {', '.join(str(seed) for seed in sorted(selected['seed'].astype(int).tolist()))}",
        "",
        "## Calibration Thresholds On CystoDS Calib",
        "",
        _to_markdown_table(threshold_summary),
        "",
        "## Frozen Validation Mean +/- Std",
        "",
        _to_markdown_table(validation_summary),
        "",
        "## EBTC Stratified Mean",
        "",
        _to_markdown_table(ebtc_class_summary),
        "",
    ]
    report = out_dir / "final_validation_cdg_cysto_v2.md"
    report.write_text("\n".join(lines), encoding="utf-8")


def _summarize(selected: pd.DataFrame, out_dir: Path) -> None:
    thresholds, validation, ebtc_by_class = _collect_outputs(selected, out_dir)
    write_table(thresholds, out_dir / "threshold_metrics_by_seed.csv")
    write_table(validation, out_dir / "frozen_validation_metrics_by_seed.csv")
    write_table(ebtc_by_class, out_dir / "ebtc_metrics_by_class_by_seed.csv")
    threshold_summary = _mean_std(
        thresholds,
        ["threshold_name"],
        ["threshold", "sensitivity", "specificity", "balanced_accuracy"],
    )
    validation_summary = _mean_std(validation, ["dataset", "threshold_name"], METRIC_COLUMNS)
    ebtc_class_summary = _mean_std(
        ebtc_by_class,
        ["threshold_name", "ebtc_class"],
        ["sensitivity", "specificity", "false_positive_rate"],
    )
    write_table(threshold_summary, out_dir / "threshold_metrics_summary.csv")
    write_table(validation_summary, out_dir / "frozen_validation_metrics_summary.csv")
    write_table(ebtc_class_summary, out_dir / "ebtc_metrics_by_class_summary.csv")
    _write_report(selected, threshold_summary, validation_summary, ebtc_class_summary, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard", default="outputs/v2/leaderboards/stage06_augmentation_leaderboard.csv")
    parser.add_argument("--model-name", default="swin_tiny")
    parser.add_argument("--variant-id", default="ce_class_balanced_sampler")
    parser.add_argument("--augmentation-name", default="modality_aware_v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2025, 3407])
    parser.add_argument("--calib-split", default="outputs/v2/splits/cystods_calib.csv")
    parser.add_argument("--internal-split", default="outputs/v2/splits/cystods_test_internal.csv")
    parser.add_argument("--external-split", default="outputs/v2/splits/ebtc_external_all.csv")
    parser.add_argument("--label-column", default="label_cancer_binary")
    parser.add_argument(
        "--out-dir",
        default="outputs/v2/final_validation/cdg_cysto_v2_swin_tiny_ce_sampler_modality_aware_v1",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_runs(args)
    write_table(selected, out_dir / "selected_stage06_runs.csv")
    for _, row in selected.iterrows():
        _run_one(row, args, out_dir)
    _summarize(selected, out_dir)
    print(f"Wrote final validation outputs to {out_dir}")


if __name__ == "__main__":
    main()
