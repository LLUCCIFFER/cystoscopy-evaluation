"""Training/evaluation engine and run artifact contract."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cdg_cysto.config.schema import EXPECTED_RUN_ARTIFACTS, save_resolved_config
from cdg_cysto.data.dataset import MetadataImageDataset
from cdg_cysto.data.transforms import build_transforms
from cdg_cysto.eval.calibration import fit_temperature_and_report
from cdg_cysto.eval.metrics import (
    build_predictions_frame,
    compute_classification_metrics,
    ensure_probabilities,
    save_confusion_matrix_plot,
    save_pr_curve_plot,
    save_reliability_diagram,
    save_roc_curve_plot,
)
from cdg_cysto.losses.class_balanced import ClassBalancedFocalLoss
from cdg_cysto.models.factory import configure_training_stage, create_model
from cdg_cysto.train.optim import build_adamw
from cdg_cysto.utils.checkpoint import load_checkpoint, save_checkpoint
from cdg_cysto.utils.device import get_device
from cdg_cysto.utils.io import ensure_dir, read_table, write_json, write_table
from cdg_cysto.utils.seed import seed_everything

TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6360000002000100ffff03000006000557bfabcf0000000049454e44ae426082"
)


def write_png_placeholder(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(TINY_PNG)


def write_run_contract(
    run_dir: str | Path,
    config: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Path]:
    """Create the required run files."""

    run_path = ensure_dir(run_dir)
    save_resolved_config(config, run_path / "config_resolved.yaml")

    epoch_frame = pd.DataFrame(
        [
            {
                "epoch": 0,
                "stage": "dry_run" if dry_run else "initialized",
                "train_loss": np.nan,
                "val_macro_f1": np.nan,
            }
        ]
    )
    write_table(epoch_frame, run_path / "metrics_epoch.csv")

    placeholder_metrics = {
        "status": "dry_run" if dry_run else "initialized",
        "macro_f1": None,
        "accuracy": None,
        "balanced_accuracy": None,
        "weighted_f1": None,
        "ece": None,
        "brier": None,
        "nll": None,
    }
    write_json(placeholder_metrics, run_path / "metrics_val_best.json")
    write_json(placeholder_metrics, run_path / "metrics_test.json")

    predictions = pd.DataFrame(columns=["sample_id", "y_true", "y_pred"])
    write_table(predictions, run_path / "predictions_test.csv")

    for name in (
        "confusion_matrix.png",
        "roc_curve.png",
        "pr_curve.png",
        "reliability_diagram.png",
    ):
        write_png_placeholder(run_path / name)

    (run_path / "checkpoint_best.pt").write_text(
        "dry-run placeholder; overwritten by real torch checkpoint\n",
        encoding="utf-8",
    )
    (run_path / "run.log").write_text(
        "CDG-Cysto run initialized\n",
        encoding="utf-8",
    )

    return {name: run_path / name for name in EXPECTED_RUN_ARTIFACTS}


def _label_column(config: dict[str, Any]) -> str:
    data_config = config.get("data", {}) or {}
    dataset_config = config.get("dataset", {}) or {}
    if "label_column" in data_config:
        return str(data_config["label_column"])
    task = config.get("task", {}) or {}
    if int(task.get("num_classes", 4)) == 2 or "binary" in str(task.get("name", "")).lower():
        return "label_cancer_binary"
    if "label_column" in dataset_config and str(dataset_config["label_column"]) != "label":
        return str(dataset_config["label_column"])
    return "label_ebtc4"


def _filter_metadata(frame: pd.DataFrame, config: dict[str, Any], split: str) -> pd.DataFrame:
    data_config = config.get("data", {}) or {}
    split_value = data_config.get(f"{split}_split", split)
    if str(split_value).lower() in {"all", "*"}:
        subset = frame
    else:
        subset = frame[frame["split"].astype(str).str.lower() == str(split_value).lower()]
    domain_key = f"{split}_domain"
    if domain_key in data_config:
        domain = str(data_config[domain_key]).upper()
        subset = subset[subset["modality"].astype(str).str.upper() == domain]
    if split == "train" and "train_domain" in data_config:
        domain = str(data_config["train_domain"]).upper()
        subset = subset[subset["modality"].astype(str).str.upper() == domain]
    if split == "test" and "test_domain" in data_config:
        domain = str(data_config["test_domain"]).upper()
        subset = subset[subset["modality"].astype(str).str.upper() == domain]
    subset = subset[subset["is_valid"].astype(bool)]
    max_samples = data_config.get(f"max_{split}_samples")
    if max_samples is not None:
        subset = subset.head(int(max_samples))
    return subset.reset_index(drop=True)


def _collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    images = [item["image"] for item in batch]
    if not torch.is_tensor(images[0]):
        tensors = []
        for image in images:
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(array).permute(2, 0, 1))
        images_tensor = torch.stack(tensors)
    else:
        images_tensor = torch.stack(images)
    return {
        "image": images_tensor,
        "target": torch.as_tensor([item["target"] for item in batch], dtype=torch.long),
        "sample_id": [item["sample_id"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }


def _build_dataloader(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    train: bool,
) -> Any:
    import torch

    task = config.get("task", {}) or {}
    model_config = config.get("model", {}) or {}
    training = config.get("training", {}) or {}
    input_size = int(model_config.get("input_size", 224))
    policy = str(model_config.get("augmentation", model_config.get("augmentation_policy", "baseline")))
    transform = build_transforms(train=train, input_size=input_size, policy=policy if train else "baseline")
    dataset = MetadataImageDataset(
        frame,
        classes=list(task.get("classes", [str(index) for index in range(int(task.get("num_classes", 2)))])),
        transform=transform,
        label_column=_label_column(config),
        patient_column="pid",
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 16)),
        shuffle=train,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=bool(training.get("pin_memory", False)),
        collate_fn=_collate_batch,
    )


def _build_loss(config: dict[str, Any], train_frame: pd.DataFrame) -> Any:
    import torch

    model_config = config.get("model", {}) or {}
    loss_config = model_config.get("loss", {}) or config.get("loss", {}) or {}
    name = str(loss_config.get("name", "ce")).lower()
    if name in {"class_balanced_focal", "cb_focal", "cb-focal"}:
        num_classes = int((config.get("task", {}) or {}).get("num_classes", 2))
        counts = train_frame[_label_column(config)].astype(int).value_counts().reindex(range(num_classes), fill_value=0)
        return ClassBalancedFocalLoss(
            counts.tolist(),
            beta=float(loss_config.get("beta", 0.999)),
            gamma=float(loss_config.get("gamma", 2.0)),
        )
    return torch.nn.CrossEntropyLoss()


def _max_batches(config: dict[str, Any], split: str) -> int | None:
    task = config.get("task", {}) or {}
    value = task.get(f"max_{split}_batches")
    return None if value is None else int(value)


def train_one_epoch(
    model: Any,
    dataloader: Any,
    criterion: Any,
    optimizer: Any,
    device: str,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:

    model.train()
    losses: list[float] = []
    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {"train_loss": float(np.mean(losses)) if losses else float("nan")}


def collect_logits(
    model: Any,
    dataloader: Any,
    device: str,
    *,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import torch

    model.eval()
    logits_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch["image"].to(device)
            logits = model(images).detach().cpu().numpy()
            logits_all.append(logits)
            targets_all.append(batch["target"].cpu().numpy())
            sample_ids.extend(batch["sample_id"])

    if not logits_all:
        return np.empty((0, 0)), np.empty((0,), dtype=int), []
    return (
        np.concatenate(logits_all, axis=0),
        np.concatenate(targets_all, axis=0),
        sample_ids,
    )


def evaluate_model(
    model: Any,
    dataloader: Any,
    device: str,
    class_names: list[str],
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    logits, targets, _ = collect_logits(model, dataloader, device, max_batches=max_batches)
    if logits.size == 0:
        return {}
    return compute_classification_metrics(targets, y_score=logits, class_names=class_names)


def _prepare_config(config: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
    prepared = dict(config)
    if "model" not in prepared and "model_config" in prepared:
        prepared["model"] = prepared["model_config"]
    if seed is not None:
        prepared["seed"] = seed
    model = dict(prepared.get("model", {}) or {})
    task = prepared.get("task", {}) or {}
    if "num_classes" in task:
        model["num_classes"] = int(task["num_classes"])
    prepared["model"] = model
    return prepared


def run_experiment(config: dict[str, Any], *, seed: int | None = None) -> dict[str, float]:
    """Run train/val/test for a resolved experiment config."""


    config = _prepare_config(config, seed)
    seed_value = int(config.get("seed", 42))
    seed_everything(seed_value)
    run_dir = ensure_dir((config.get("outputs", {}) or {}).get("run_dir", "outputs/runs/unnamed"))
    save_resolved_config(config, run_dir / "config_resolved.yaml")

    dataset_config = config.get("dataset", {}) or {}
    external_dataset_config = config.get("external_dataset", {}) or {}
    metadata_csv = Path(dataset_config.get("metadata_csv", "data/processed/EBTC/metadata.csv"))
    metadata = read_table(metadata_csv)
    if external_dataset_config:
        test_metadata_csv = Path(external_dataset_config.get("metadata_csv", metadata_csv))
        test_metadata = read_table(test_metadata_csv)
    else:
        test_metadata = metadata
    train_frame = _filter_metadata(metadata, config, "train")
    val_frame = _filter_metadata(metadata, config, "val")
    test_frame = _filter_metadata(test_metadata, config, "test")
    if train_frame.empty or val_frame.empty or test_frame.empty:
        msg = (
            "Train/val/test split is empty after filtering: "
            f"train={len(train_frame)}, val={len(val_frame)}, test={len(test_frame)}"
        )
        raise ValueError(msg)

    device = get_device(prefer_cuda=bool((config.get("training", {}) or {}).get("prefer_cuda", True)))
    class_names = list((config.get("task", {}) or {}).get("classes", []))
    model = create_model(config["model"]).to(device)
    criterion = _build_loss(config, train_frame).to(device)

    train_loader = _build_dataloader(train_frame, config, train=True)
    val_loader = _build_dataloader(val_frame, config, train=False)
    test_loader = _build_dataloader(test_frame, config, train=False)

    training = config.get("training", {}) or {}
    epoch_rows: list[dict[str, float | int | str]] = []
    best_macro_f1 = -math.inf
    best_metrics: dict[str, float] = {}
    checkpoint_path = run_dir / "checkpoint_best.pt"
    global_epoch = 0

    stages = [
        ("stage1", int(training.get("epochs_stage1", training.get("epochs", 1))), 1),
        ("stage2", int(training.get("epochs_stage2", 0)), 2),
    ]
    for stage_name, epochs, stage_number in stages:
        if epochs <= 0:
            continue
        configure_training_stage(
            model,
            stage_number,
            unfreeze_last_blocks=int(
                ((config.get("model", {}) or {}).get("training", {}) or {}).get(
                    "stage2_unfreeze_last_blocks", 2
                )
            ),
        )
        optimizer = build_adamw(
            model,
            learning_rate_head=float(training.get("learning_rate_head", 3e-4)),
            learning_rate_backbone=float(training.get("learning_rate_backbone", 1e-5)),
            weight_decay=float(training.get("weight_decay", 0.05)),
        )
        for _ in range(epochs):
            global_epoch += 1
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                max_batches=_max_batches(config, "train"),
            )
            val_metrics = evaluate_model(
                model,
                val_loader,
                device,
                class_names,
                max_batches=_max_batches(config, "val"),
            )
            row: dict[str, float | int | str] = {
                "epoch": global_epoch,
                "stage": stage_name,
                **train_metrics,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            epoch_rows.append(row)
            write_table(pd.DataFrame(epoch_rows), run_dir / "metrics_epoch.csv")
            val_macro_f1 = float(val_metrics.get("macro_f1", float("nan")))
            if np.isfinite(val_macro_f1) and val_macro_f1 >= best_macro_f1:
                best_macro_f1 = val_macro_f1
                best_metrics = val_metrics
                save_checkpoint(
                    {
                        "model_state": model.state_dict(),
                        "config": config,
                        "epoch": global_epoch,
                        "seed": seed_value,
                        "val_metrics": val_metrics,
                    },
                    checkpoint_path,
                )
                write_json(best_metrics, run_dir / "metrics_val_best.json")

    if not checkpoint_path.exists():
        save_checkpoint(
            {"model_state": model.state_dict(), "config": config, "epoch": 0, "seed": seed_value},
            checkpoint_path,
        )
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    val_logits, val_targets, _ = collect_logits(
        model, val_loader, device, max_batches=_max_batches(config, "val")
    )
    test_logits, test_targets, test_sample_ids = collect_logits(
        model, test_loader, device, max_batches=_max_batches(config, "test")
    )
    test_prob = ensure_probabilities(test_logits)
    test_metrics = compute_classification_metrics(test_targets, y_score=test_prob, class_names=class_names)

    calibration_config = ((config.get("model", {}) or {}).get("calibration", {}) or {})
    if bool(calibration_config.get("enabled", False)) and val_logits.size and test_logits.size:
        scaler, calibration_report = fit_temperature_and_report(
            val_logits,
            val_targets,
            test_logits,
            test_targets,
        )
        test_prob = scaler.predict_proba(test_logits)
        test_metrics.update(calibration_report)

    predictions = build_predictions_frame(test_sample_ids, test_targets, test_prob)
    write_table(predictions, run_dir / "predictions_test.csv")
    write_json(test_metrics, run_dir / "metrics_test.json")
    save_confusion_matrix_plot(
        test_targets,
        predictions["y_pred"].to_numpy(),
        class_names,
        run_dir / "confusion_matrix.png",
    )
    save_roc_curve_plot(test_targets, test_prob, run_dir / "roc_curve.png")
    save_pr_curve_plot(test_targets, test_prob, run_dir / "pr_curve.png")
    save_reliability_diagram(test_targets, test_prob, run_dir / "reliability_diagram.png")
    (run_dir / "run.log").write_text(
        f"Completed seed={seed_value} on device={device}\n"
        f"best_val_macro_f1={best_macro_f1}\n"
        f"test_macro_f1={test_metrics.get('macro_f1')}\n",
        encoding="utf-8",
    )
    return test_metrics


def summarize_seed_runs(seed_metrics: list[dict[str, float]], output_csv: str | Path) -> pd.DataFrame:
    """Write mean/std summary for multiple seeds."""

    rows: list[dict[str, float | str]] = []
    frame = pd.DataFrame(seed_metrics)
    numeric = frame.select_dtypes(include=[np.number])
    for column in numeric.columns:
        rows.append({"metric": column, "mean": numeric[column].mean(), "std": numeric[column].std(ddof=0)})
    summary = pd.DataFrame(rows)
    write_table(summary, output_csv)
    return summary


def run_v2_experiment(config: dict[str, Any], *, command: str = "") -> dict[str, float]:
    """Run the CDG-Cysto v2 engine without coupling it to the v1 training path."""

    from cdg_cysto.train.engine_v2 import run_v2_experiment as _run_v2_experiment

    return _run_v2_experiment(config, command=command)
