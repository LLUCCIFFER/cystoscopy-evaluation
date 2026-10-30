"""Create the Stage 06 lightweight augmentation config from stable Stage 05 top candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdg_cysto.config.schema import save_resolved_config  # noqa: E402


def _loss_config(row: pd.Series) -> dict[str, Any]:
    loss = {"name": str(row["loss_name"])}
    tau = row.get("loss_tau", "")
    if pd.notna(tau) and str(tau) not in {"", "nan"}:
        loss["tau"] = float(tau)
    return loss


def _sampler_config(row: pd.Series) -> dict[str, Any]:
    sampler_name = str(row["sampler_name"])
    return {
        "name": sampler_name,
        "enabled": sampler_name not in {"none", "false", "off"},
        "mode": "inverse_frequency",
    }


def _candidate(row: pd.Series) -> dict[str, Any]:
    return {
        "model": str(row["model_name"]),
        "variant_id": str(row["variant_id"]),
        "loss": _loss_config(row),
        "sampler": _sampler_config(row),
    }


def _score_column(frame: pd.DataFrame) -> str:
    for column in ("selection_score_v2_mean", "val_selection_score_mean", "val_auprc_mean"):
        if column in frame.columns:
            return column
    msg = "Stage 05 summary does not contain a usable ranking score column."
    raise ValueError(msg)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="outputs/v2/leaderboards/stage05_multiseed_summary.csv")
    parser.add_argument("--output", default="configs/v2/experiment/stage06_augmentation_top2.yaml")
    parser.add_argument("--top-n", type=int, default=2)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    if summary.empty:
        msg = "Stage 05 summary is empty; cannot create Stage 06 config."
        raise ValueError(msg)
    if "unstable" in summary.columns:
        stable = summary[~summary["unstable"].map(_truthy)].copy()
    elif "stability_status" in summary.columns:
        stable = summary[summary["stability_status"].astype(str) == "STABLE"].copy()
    else:
        stable = summary.copy()
    if "n_seeds" not in stable.columns:
        msg = "Stage 05 summary must contain n_seeds before Stage 06."
        raise ValueError(msg)
    stable = stable[stable["n_seeds"].astype(int) >= 3].copy()
    if len(stable) < args.top_n:
        msg = f"Need {args.top_n} stable candidates with at least 3 seeds, found {len(stable)}."
        raise ValueError(msg)

    score_column = _score_column(stable)
    selected = stable.sort_values(score_column, ascending=False).head(args.top_n)
    config = {
        "experiment_name": "cdg_cysto_v2_stage06_augmentation",
        "run_type": "formal",
        "stage": "stage06_augmentation",
        "seed": 42,
        "task": {
            "name": "cancer_binary",
            "label_column": "label_cancer_binary",
            "num_classes": 2,
            "classes": ["non_cancer", "cancer"],
        },
        "dataset": {
            "train_split": "outputs/v2/splits/cystods_train.csv",
            "val_split": "outputs/v2/splits/cystods_val.csv",
            "calib_split": "outputs/v2/splits/cystods_calib.csv",
            "test_internal_split": "outputs/v2/splits/cystods_test_internal.csv",
            "external_split": "outputs/v2/splits/ebtc_external_all.csv",
        },
        "training": {
            "epochs": 30,
            "early_stopping_patience": 8,
            "batch_size": 16,
            "optimizer": "adamw",
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "monitor": "val_auprc",
            "seeds": [42, 2025, 3407],
            "num_workers": 0,
        },
        "augmentation_grid": [
            {"name": "basic", "policy": "basic"},
            {"name": "modality_aware_v1", "policy": "modality_aware_v1"},
        ],
        "evaluation": {
            "threshold": 0.5,
            "save_logits": True,
            "save_predictions": True,
            "compute_threshold_sweep": True,
        },
        "outputs": {"root_dir": "outputs/v2/runs"},
        "candidate_variants": [_candidate(row) for _, row in selected.iterrows()],
    }
    save_resolved_config(config, args.output)
    print(selected[["model_name", "variant_id", score_column]].to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
