"""Run a CDG-Cysto v2 YAML grid over candidate backbones."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdg_cysto.config.schema import load_yaml, save_resolved_config  # noqa: E402
from cdg_cysto.train.engine_v2 import run_v2_experiment  # noqa: E402
from cdg_cysto.utils.io import ensure_dir, read_json, write_table  # noqa: E402


def _run_name(
    base_name: str,
    model_config: dict[str, Any],
    seed: int,
    variant_slug: str = "",
    augmentation_slug: str = "",
) -> str:
    model_id = str(model_config.get("model_id", "model"))
    name = str(model_config.get("name", "model"))
    variant = f"_{variant_slug}" if variant_slug else ""
    augmentation = f"_{augmentation_slug}" if augmentation_slug else ""
    return f"{base_name}_{model_id}_{name}{variant}{augmentation}_seed{seed}"


def _safe_slug(value: str) -> str:
    keep = [character.lower() if character.isalnum() else "_" for character in value]
    slug = "_".join(part for part in "".join(keep).split("_") if part)
    return slug or "variant"


def _variant_slug(config: dict[str, Any]) -> str:
    if config.get("variant_id"):
        return _safe_slug(str(config["variant_id"]))
    imbalance = config.get("imbalance", {}) or {}
    loss = imbalance.get("loss", {}) or {}
    sampler = imbalance.get("sampler", {}) or {}
    loss_name = str(loss.get("name", "cross_entropy"))
    sampler_name = str(sampler.get("name", "none" if not sampler.get("enabled") else "sampler"))
    tau = loss.get("tau")
    tau_part = f"_tau_{tau}" if tau is not None else ""
    return _safe_slug(f"{loss_name}{tau_part}_{sampler_name}")


def _model_config_from_name(name: str) -> dict[str, Any]:
    mapping = {
        "efficientnet_b3": {
            "model_id": "M0",
            "name": "efficientnet_b3",
            "source": "timm",
            "backbone": "efficientnet_b3",
            "pretrained": True,
            "input_size": 224,
        },
        "swin_tiny": {
            "model_id": "M1",
            "name": "swin_tiny",
            "source": "timm",
            "backbone": "swin_tiny_patch4_window7_224",
            "pretrained": True,
            "input_size": 224,
        },
        "convnext_tiny": {
            "model_id": "M2",
            "name": "convnext_tiny",
            "source": "timm",
            "backbone": "convnext_tiny",
            "pretrained": True,
            "input_size": 224,
        },
        "resnet50": {
            "model_id": "M3",
            "name": "resnet50",
            "source": "timm",
            "backbone": "resnet50",
            "pretrained": True,
            "input_size": 224,
        },
        "dinov2_linear": {
            "model_id": "M4",
            "name": "dinov2_linear",
            "source": "torch_hub",
            "backbone": "dinov2_vits14",
            "pretrained": True,
            "input_size": 224,
            "feature_dim": 384,
        },
    }
    if name not in mapping:
        msg = f"Unsupported v2 model shortcut: {name}"
        raise ValueError(msg)
    return mapping[name]


def normalize_grid_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    normalized.setdefault("run_type", "formal")
    dataset = normalized.get("dataset", {}) or {}
    if dataset:
        normalized["data"] = {
            "train_csv": dataset.get("train_split"),
            "val_csv": dataset.get("val_split"),
            "calib_csv": dataset.get("calib_split"),
            "test_internal_csv": dataset.get("test_internal_split"),
            "external_csv": dataset.get("external_split"),
        }
    training = normalized.get("training", {}) or {}
    if "early_stopping_patience" in training and "patience" not in training:
        training["patience"] = training["early_stopping_patience"]
    if "lr" in training and "learning_rate" not in training:
        training["learning_rate"] = training["lr"]
    if "seeds" in training and "seed" not in normalized:
        normalized["seed"] = int(training["seeds"][0])
    normalized["training"] = training

    loss = normalized.get("loss", {}) or {}
    sampler = normalized.get("sampler", {}) or {}
    augmentation = normalized.get("augmentation", {}) or {}
    normalized["imbalance"] = normalized.get("imbalance", {}) or {}
    normalized["imbalance"]["loss"] = normalized["imbalance"].get("loss", {}) or {
        "name": loss.get("name", "cross_entropy")
    }
    sampler_name = str(sampler.get("name", "none")).lower()
    normalized["imbalance"]["sampler"] = normalized["imbalance"].get("sampler", {}) or {
        "enabled": sampler_name not in {"none", "false", "off"},
        "mode": sampler.get("mode", "inverse_frequency"),
        "name": sampler_name,
    }
    if "policy" not in augmentation:
        augmentation["policy"] = augmentation.get("name", "basic")
    normalized["augmentation"] = augmentation

    if "candidate_pool" not in normalized and "models" in normalized:
        normalized["candidate_pool"] = [
            _model_config_from_name(model) if isinstance(model, str) else model
            for model in normalized["models"]
        ]
    return normalized


def _normalized_imbalance_variant(variant: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(variant)
    loss = normalized.get("loss", {}) or {}
    sampler = normalized.get("sampler", {}) or {}
    if "name" in normalized and not loss:
        loss = {"name": normalized["name"]}
    sampler_name = str(sampler.get("name", "none")).lower()
    normalized["loss"] = loss or {"name": "cross_entropy"}
    normalized["sampler"] = sampler or {
        "enabled": sampler_name not in {"none", "false", "off"},
        "mode": "inverse_frequency",
        "name": sampler_name,
    }
    return normalized


def _normalized_augmentation_variant(variant: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(variant, str):
        return {"name": variant, "policy": variant}
    normalized = copy.deepcopy(variant)
    if "policy" not in normalized:
        normalized["policy"] = normalized.get("name", "basic")
    if "name" not in normalized:
        normalized["name"] = normalized.get("policy", "basic")
    return normalized


def _augmentation_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    variants = list(config.get("augmentation_grid", []) or [])
    if variants:
        return [_normalized_augmentation_variant(variant) for variant in variants]
    return [_normalized_augmentation_variant(config.get("augmentation", {}) or {"name": "basic"})]


def expand_grid(config: dict[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    config = normalize_grid_config(config)
    augmentation_variants = _augmentation_variants(config)
    use_augmentation_slug = len(augmentation_variants) > 1
    candidate_variants = list(config.get("candidate_variants", []) or [])
    if candidate_variants:
        runs: list[dict[str, Any]] = []
        base_name = str(config.get("experiment_name", "cdg_cysto_v2"))
        root_dir = Path((config.get("outputs", {}) or {}).get("root_dir", "outputs/v2/runs"))
        for seed in seeds:
            for candidate in candidate_variants:
                for augmentation_variant in augmentation_variants:
                    model_value = candidate.get("model", candidate.get("model_name"))
                    model_config = (
                        _model_config_from_name(model_value)
                        if isinstance(model_value, str)
                        else dict(model_value or {})
                    )
                    if not model_config:
                        msg = "Each candidate_variants item must define model or model_name."
                        raise ValueError(msg)
                    imbalance_variant = _normalized_imbalance_variant(candidate)
                    run_config = copy.deepcopy(config)
                    run_config.pop("candidate_variants", None)
                    run_config.pop("candidate_pool", None)
                    run_config.pop("models", None)
                    run_config.pop("imbalance_grid", None)
                    run_config.pop("variants", None)
                    run_config.pop("augmentation_grid", None)
                    run_config["seed"] = seed
                    run_config["model"] = model_config
                    run_config["augmentation"] = copy.deepcopy(augmentation_variant)
                    run_config["imbalance"] = {
                        "loss": copy.deepcopy(imbalance_variant["loss"]),
                        "sampler": copy.deepcopy(imbalance_variant["sampler"]),
                    }
                    if imbalance_variant.get("variant_id"):
                        run_config["variant_id"] = str(imbalance_variant["variant_id"])
                    augmentation_slug = (
                        _safe_slug(str(augmentation_variant.get("name", "basic")))
                        if use_augmentation_slug
                        else ""
                    )
                    run_config["outputs"] = dict(run_config.get("outputs", {}) or {})
                    run_config["outputs"]["run_dir"] = str(
                        root_dir
                        / _run_name(
                            base_name,
                            model_config,
                            seed,
                            _variant_slug(run_config),
                            augmentation_slug,
                        )
                    )
                    runs.append(run_config)
        return runs

    candidates = list(config.get("candidate_pool", []))
    if not candidates and "model" in config:
        candidates = [config["model"]]
    if not candidates:
        msg = "v2 grid config must contain candidate_pool or model."
        raise ValueError(msg)
    runs: list[dict[str, Any]] = []
    base_name = str(config.get("experiment_name", "cdg_cysto_v2"))
    root_dir = Path((config.get("outputs", {}) or {}).get("root_dir", "outputs/v2/runs"))
    imbalance_variants = [
        _normalized_imbalance_variant(variant)
        for variant in list(config.get("imbalance_grid", []) or config.get("variants", []) or [])
    ]
    if not imbalance_variants:
        default_imbalance = copy.deepcopy(config.get("imbalance", {}) or {})
        default_imbalance.setdefault("variant_id", str(config.get("variant_id", "")))
        imbalance_variants = [_normalized_imbalance_variant(default_imbalance)]
    for seed in seeds:
        for model_config in candidates:
            for imbalance_variant in imbalance_variants:
                for augmentation_variant in augmentation_variants:
                    run_config = copy.deepcopy(config)
                    run_config.pop("candidate_pool", None)
                    run_config.pop("models", None)
                    run_config.pop("imbalance_grid", None)
                    run_config.pop("variants", None)
                    run_config.pop("augmentation_grid", None)
                    run_config["seed"] = seed
                    run_config["model"] = model_config
                    run_config["augmentation"] = copy.deepcopy(augmentation_variant)
                    run_config["imbalance"] = {
                        "loss": copy.deepcopy(imbalance_variant["loss"]),
                        "sampler": copy.deepcopy(imbalance_variant["sampler"]),
                    }
                    if imbalance_variant.get("variant_id"):
                        run_config["variant_id"] = str(imbalance_variant["variant_id"])
                    augmentation_slug = (
                        _safe_slug(str(augmentation_variant.get("name", "basic")))
                        if use_augmentation_slug
                        else ""
                    )
                    run_config["outputs"] = dict(run_config.get("outputs", {}) or {})
                    run_config["outputs"]["run_dir"] = str(
                        root_dir
                        / _run_name(
                            base_name,
                            model_config,
                            seed,
                            _variant_slug(run_config),
                            augmentation_slug,
                        )
                    )
                    runs.append(run_config)
    return runs


def _run_metadata(row_config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else {}
    imbalance = row_config.get("imbalance", {}) or {}
    loss = imbalance.get("loss", {}) or {}
    sampler = imbalance.get("sampler", {}) or {}
    augmentation = row_config.get("augmentation", {}) or {}
    return {
        "run_id": manifest.get("run_id", run_dir.name),
        "run_type": row_config.get("run_type", "formal"),
        "stage": row_config.get("stage", ""),
        "task": (row_config.get("task", {}) or {}).get("name", ""),
        "variant_id": row_config.get("variant_id", _variant_slug(row_config)),
        "loss_name": loss.get("name", "cross_entropy"),
        "loss_tau": loss.get("tau", ""),
        "sampler_name": sampler.get("name", "none")
        if "name" in sampler
        else ("enabled" if sampler.get("enabled") else "none"),
        "augmentation_name": augmentation.get("name", augmentation.get("policy", "basic")),
        "monitor": (row_config.get("training", {}) or {}).get("monitor", "val_selection_score"),
    }


def _write_smoke_report(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 03 Smoke Report",
        "",
        "Smoke runs are connectivity checks only and are excluded from formal candidate selection.",
        "",
        "| model | selection_score | AUROC | Macro-F1 | sensitivity_cancer | specificity_cancer | ECE |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['model_name']} | {row.get('val_selection_score', np.nan):.6f} | "
            f"{row.get('val_auroc', np.nan):.6f} | {row.get('val_macro_f1', np.nan):.6f} | "
            f"{row.get('val_sensitivity_cancer', np.nan):.6f} | "
            f"{row.get('val_specificity_cancer', np.nan):.6f} | {row.get('val_ece', np.nan):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith("prob_")],
        key=lambda value: int(value.split("_")[1]),
    )


def write_stage03_figures(frame: pd.DataFrame, output_dir: str | Path = "outputs/v2/figures") -> None:
    output = ensure_dir(output_dir)
    if frame.empty:
        return

    fig_roc, ax_roc = plt.subplots(figsize=(5, 4), dpi=160)
    fig_pr, ax_pr = plt.subplots(figsize=(5, 4), dpi=160)
    fig_cm, axes_cm = plt.subplots(1, len(frame), figsize=(4 * len(frame), 4), dpi=160)
    axes = np.atleast_1d(axes_cm)

    for axis, (_, row) in zip(axes, frame.iterrows(), strict=False):
        predictions_path = Path(str(row["run_dir"])) / "predictions_val.csv"
        if not predictions_path.exists():
            continue
        predictions = pd.read_csv(predictions_path)
        prob_columns = _probability_columns(predictions)
        if len(prob_columns) < 2:
            continue
        y_true = predictions["y_true"].to_numpy(dtype=int)
        y_pred = predictions["y_pred"].to_numpy(dtype=int)
        scores = predictions["prob_1"].to_numpy(dtype=float)
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            ax_roc.plot(fpr, tpr, label=str(row["model_name"]))
            precision, recall, _ = precision_recall_curve(y_true, scores)
            ax_pr.plot(recall, precision, label=str(row["model_name"]))
        except ValueError:
            pass
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        ConfusionMatrixDisplay(matrix, display_labels=["non_cancer", "cancer"]).plot(
            ax=axis,
            colorbar=False,
        )
        axis.set_title(str(row["model_name"]))

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(fontsize=8)
    fig_roc.tight_layout()
    fig_roc.savefig(output / "stage03_roc_curves.png")
    plt.close(fig_roc)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(fontsize=8)
    fig_pr.tight_layout()
    fig_pr.savefig(output / "stage03_pr_curves.png")
    plt.close(fig_pr)

    fig_cm.tight_layout()
    fig_cm.savefig(output / "stage03_confusion_matrices.png")
    plt.close(fig_cm)


def _leaderboard_name(stage: str) -> str:
    lowered = stage.lower()
    if "stage06" in lowered and "augmentation" in lowered:
        return "stage06_augmentation_leaderboard.csv"
    if "stage05" in lowered and "multiseed" in lowered:
        return "stage05_multiseed_leaderboard.csv"
    if "stage04" in lowered and "imbalance" in lowered:
        return "stage04_imbalance_leaderboard.csv"
    if "stage04" in lowered and "augmentation" in lowered:
        return "stage04_augmentation_leaderboard.csv"
    if "smoke" in lowered:
        return "stage03_smoke_leaderboard.csv"
    return "stage03_backbone_leaderboard.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v2/primary_cancer_grid.yaml")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = normalize_grid_config(load_yaml(args.config))
    default_seed = int(config.get("seed", 42))
    training_seeds = (config.get("training", {}) or {}).get("seeds", [default_seed])
    seeds = (
        [int(value) for value in training_seeds]
        if not args.seeds
        else [int(value) for value in args.seeds.split(",")]
    )
    runs = expand_grid(config, seeds)
    leaderboard_rows: list[dict[str, Any]] = []
    for run_config in runs:
        run_dir = ensure_dir((run_config.get("outputs", {}) or {})["run_dir"])
        save_resolved_config(run_config, run_dir / "config_resolved.yaml")
        if args.dry_run:
            print(f"Dry run: {run_dir}")
            continue
        command = " ".join(sys.argv)
        metrics = run_v2_experiment(run_config, command=command)
        row = {
            "run_dir": str(run_dir),
            "seed": run_config["seed"],
            "model_id": run_config["model"].get("model_id", ""),
            "model_name": run_config["model"].get("name", ""),
        }
        row.update(_run_metadata(run_config, run_dir))
        row.update({f"val_{key}": value for key, value in metrics.items()})
        leaderboard_rows.append(row)
        print(f"Completed {run_dir}: val_selection_score={metrics.get('selection_score')}")

    if leaderboard_rows:
        leaderboard_dir = ensure_dir("outputs/v2/leaderboards")
        frame = pd.DataFrame(leaderboard_rows)
        smoke = frame[frame["run_type"].astype(str) == "smoke"].copy()
        formal = frame[frame["run_type"].astype(str) == "formal"].copy()
        if not smoke.empty:
            write_table(smoke, leaderboard_dir / "stage03_smoke_leaderboard.csv")
            _write_smoke_report(smoke, Path("outputs/v2/reports/stage03_smoke_report.md"))
        if not formal.empty:
            stage = str(formal["stage"].iloc[0]) if "stage" in formal.columns else ""
            leaderboard_name = _leaderboard_name(stage)
            write_table(formal, leaderboard_dir / leaderboard_name)
            if leaderboard_name == "stage03_backbone_leaderboard.csv":
                write_stage03_figures(formal)
        print(f"Wrote leaderboard rows: {len(frame)}")


if __name__ == "__main__":
    main()
