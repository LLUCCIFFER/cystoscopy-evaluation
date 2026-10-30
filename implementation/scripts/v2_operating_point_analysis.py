"""Operating-point trade-off analysis for the frozen CDG-Cysto v2 model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdg_cysto.eval.thresholding import binary_confusion_at_threshold  # noqa: E402
from cdg_cysto.utils.io import ensure_dir, write_table  # noqa: E402


def _logit_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith("logit_")],
        key=lambda value: int(value.split("_")[1]),
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _threshold_sweep(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in np.linspace(0.0, 1.0, 101):
        y_pred = (scores >= threshold).astype(int)
        confusion = binary_confusion_at_threshold(labels, scores, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "sensitivity": confusion["sensitivity"],
                "specificity": confusion["specificity"],
                "ppv": confusion["ppv"],
                "npv": confusion["npv"],
                "balanced_accuracy": (confusion["sensitivity"] + confusion["specificity"]) / 2,
                "macro_f1": float(f1_score(labels, y_pred, average="macro", zero_division=0)),
                "youden_index": confusion["sensitivity"] + confusion["specificity"] - 1.0,
                "predicted_positive_rate": float(y_pred.mean()),
                "tp": confusion["tp"],
                "tn": confusion["tn"],
                "fp": confusion["fp"],
                "fn": confusion["fn"],
            }
        )
    return pd.DataFrame(rows)


def _mean_std(frame: pd.DataFrame, group_columns: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, Any] = dict(zip(group_columns, key_tuple, strict=False))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group.columns else len(group)
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean_std"] = (
                f"{row[f'{metric}_mean']:.3f} +/- {row[f'{metric}_std']:.3f}"
                if np.isfinite(row[f"{metric}_mean"])
                else ""
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _read_thresholds(seed_dir: Path) -> dict[str, float]:
    frame = pd.read_csv(seed_dir / "threshold_metrics.csv")
    return dict(zip(frame["threshold_name"].astype(str), frame["threshold"].astype(float), strict=False))


def _load_scores(seed_dir: Path, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    logits = pd.read_csv(seed_dir / dataset / "logits_calibrated.csv")
    labels = logits["y_true"].to_numpy(dtype=int)
    probabilities = _softmax(logits[_logit_columns(logits)].to_numpy(dtype=float))
    return labels, probabilities[:, 1]


def _plot_tradeoff(summary: pd.DataFrame, dataset: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    data = summary[summary["dataset"].eq(dataset)].sort_values("threshold")
    plt.figure(figsize=(8, 5))
    for metric, label in [
        ("sensitivity_mean", "Sensitivity"),
        ("specificity_mean", "Specificity"),
        ("ppv_mean", "PPV"),
        ("npv_mean", "NPV"),
    ]:
        plt.plot(data["threshold"], data[metric], label=label, linewidth=2)
    plt.xlabel("Cancer probability threshold")
    plt.ylabel("Metric")
    plt.title(f"Operating point trade-off: {dataset}")
    plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def _write_report(output_dir: Path, named_summary: pd.DataFrame) -> None:
    lines = [
        "# CDG-Cysto v2 Operating Point Analysis",
        "",
        "Status: PASS",
        "",
        "This diagnostic summarizes calibrated B3 CDG-Cysto v2 predictions across probability thresholds. It is intended to show the sensitivity-specificity and PPV-NPV trade-off, not to select a new threshold.",
        "",
        "## Named Operating Points",
        "",
        "| dataset | threshold | sensitivity | specificity | PPV | NPV | macro-F1 | predicted positive rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in named_summary.sort_values(["dataset", "threshold_name"]).iterrows():
        lines.append(
            f"| {row['dataset']} | {row['threshold_name']} | "
            f"{row['sensitivity_mean']:.3f} +/- {row['sensitivity_std']:.3f} | "
            f"{row['specificity_mean']:.3f} +/- {row['specificity_std']:.3f} | "
            f"{row['ppv_mean']:.3f} +/- {row['ppv_std']:.3f} | "
            f"{row['npv_mean']:.3f} +/- {row['npv_std']:.3f} | "
            f"{row['macro_f1_mean']:.3f} +/- {row['macro_f1_std']:.3f} | "
            f"{row['predicted_positive_rate_mean']:.3f} +/- {row['predicted_positive_rate_std']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: the sensitivity-oriented operating point increases cancer detection but also increases false positives, which is reflected by lower specificity and higher predicted-positive rate.",
        ]
    )
    (output_dir / "operating_point_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-dir", default="outputs/v2/final_validation/frozen_baselines/B3_CDG_Cysto_v2")
    parser.add_argument("--output-dir", default="outputs/v2/operating_points")
    args = parser.parse_args()

    method_dir = Path(args.method_dir)
    output_dir = ensure_dir(args.output_dir)
    sweep_rows: list[pd.DataFrame] = []
    named_rows: list[dict[str, Any]] = []
    datasets = ["cystods_internal", "ebtc_external"]
    metrics = [
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "balanced_accuracy",
        "macro_f1",
        "youden_index",
        "predicted_positive_rate",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    for seed_dir in sorted(method_dir.glob("seed_*")):
        seed = int(seed_dir.name.split("_", maxsplit=1)[1])
        thresholds = _read_thresholds(seed_dir)
        for dataset in datasets:
            labels, scores = _load_scores(seed_dir, dataset)
            sweep = _threshold_sweep(labels, scores)
            sweep.insert(0, "dataset", dataset)
            sweep.insert(1, "seed", seed)
            sweep_rows.append(sweep)
            for threshold_name, threshold in thresholds.items():
                y_pred = (scores >= threshold).astype(int)
                confusion = binary_confusion_at_threshold(labels, scores, threshold)
                named_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "threshold_name": threshold_name,
                        "threshold": threshold,
                        "sensitivity": confusion["sensitivity"],
                        "specificity": confusion["specificity"],
                        "ppv": confusion["ppv"],
                        "npv": confusion["npv"],
                        "balanced_accuracy": (confusion["sensitivity"] + confusion["specificity"]) / 2,
                        "macro_f1": float(f1_score(labels, y_pred, average="macro", zero_division=0)),
                        "youden_index": confusion["sensitivity"] + confusion["specificity"] - 1.0,
                        "predicted_positive_rate": float(y_pred.mean()),
                        "tp": confusion["tp"],
                        "tn": confusion["tn"],
                        "fp": confusion["fp"],
                        "fn": confusion["fn"],
                    }
                )

    sweep_frame = pd.concat(sweep_rows, ignore_index=True)
    named_frame = pd.DataFrame(named_rows)
    sweep_summary = _mean_std(sweep_frame, ["dataset", "threshold"], metrics)
    named_summary = _mean_std(named_frame, ["dataset", "threshold_name"], ["threshold", *metrics])
    write_table(sweep_frame, output_dir / "threshold_sweep_by_seed.csv")
    write_table(sweep_summary, output_dir / "threshold_sweep_summary.csv")
    write_table(named_frame, output_dir / "named_operating_points_by_seed.csv")
    write_table(named_summary, output_dir / "named_operating_points_summary.csv")
    for dataset in datasets:
        _plot_tradeoff(
            sweep_summary,
            dataset,
            output_dir / f"threshold_tradeoff_{dataset}.png",
        )
    _write_report(output_dir, named_summary)
    print(output_dir / "operating_point_analysis.md")


if __name__ == "__main__":
    main()
