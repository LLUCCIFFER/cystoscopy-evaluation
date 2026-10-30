"""CDG-Cysto v2 training engine."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cdg_cysto.augmentations.modality_aware import (
    build_modality_aware_transform,
    metrics_by_modality,
)
from cdg_cysto.data.transforms import build_transforms
from cdg_cysto.eval.metrics import (
    build_predictions_frame,
    compute_classification_metrics,
    ensure_probabilities,
)
from cdg_cysto.losses.balanced_softmax import BalancedSoftmaxLoss
from cdg_cysto.losses.logit_adjustment import LogitAdjustmentLoss
from cdg_cysto.models.factory import create_model
from cdg_cysto.samplers.class_balanced_sampler import (
    build_class_balanced_sampler,
    class_counts,
)
from cdg_cysto.utils.checkpoint import save_checkpoint
from cdg_cysto.utils.device import get_device
from cdg_cysto.utils.io import ensure_dir, write_json, write_table
from cdg_cysto.utils.manifest_v2 import write_run_manifest
from cdg_cysto.utils.seed import seed_everything


class V2MetadataDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        label_column: str,
        transform: Any,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.label_column = label_column
        self.transform = transform
        self.targets = self.frame[label_column].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        sample = row.to_dict()
        if getattr(self.transform, "uses_sample", False):
            transformed = self.transform(image=np.asarray(image), sample=sample)
        else:
            transformed = self.transform(image=np.asarray(image))
        image_tensor = transformed["image"] if isinstance(transformed, dict) else transformed
        return {
            "image": image_tensor,
            "target": int(row[self.label_column]),
            "sample_id": str(row["sample_id"]),
            "modality": str(row.get("modality", "")),
        }


def collate_v2(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    images = torch.stack([item["image"] for item in batch])
    return {
        "image": images,
        "target": torch.as_tensor([item["target"] for item in batch], dtype=torch.long),
        "sample_id": [item["sample_id"] for item in batch],
        "modality": [item["modality"] for item in batch],
    }


def _build_transform(config: dict[str, Any], *, train: bool) -> Any:
    model_config = config.get("model", {}) or {}
    aug_config = config.get("augmentation", {}) or {}
    policy = str(aug_config.get("policy", "basic"))
    input_size = int(model_config.get("input_size", 224))
    if policy.startswith("modality_aware") or policy == "basic":
        return build_modality_aware_transform(policy=policy, train=train, input_size=input_size)
    return build_transforms(train=train, input_size=input_size, policy=policy)


def _build_loader(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    train: bool,
    sampler: Any = None,
) -> Any:
    import torch

    training = config.get("training", {}) or {}
    task = config.get("task", {}) or {}
    dataset = V2MetadataDataset(
        frame,
        label_column=str(task.get("label_column", "label_cancer_binary")),
        transform=_build_transform(config, train=train),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 16)),
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=int(training.get("num_workers", 0)),
        collate_fn=collate_v2,
    )


def _limit_frame_stratified(
    frame: pd.DataFrame,
    max_samples: int | None,
    *,
    label_column: str,
    seed: int,
) -> pd.DataFrame:
    if max_samples is None or len(frame) <= max_samples:
        return frame.reset_index(drop=True)
    fractions = frame[label_column].value_counts(normalize=True)
    parts: list[pd.DataFrame] = []
    remaining = int(max_samples)
    labels = list(fractions.index)
    for label in labels[:-1]:
        n = max(1, int(round(max_samples * float(fractions.loc[label]))))
        subset = frame[frame[label_column] == label].sample(
            n=min(n, int((frame[label_column] == label).sum())),
            random_state=seed + int(label),
        )
        parts.append(subset)
        remaining -= len(subset)
    last_label = labels[-1]
    last_pool = frame[frame[label_column] == last_label]
    parts.append(last_pool.sample(n=min(max(remaining, 1), len(last_pool)), random_state=seed + 99))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _selection_score(metrics: dict[str, float]) -> float:
    return float(
        0.25 * metrics.get("auroc", 0.0)
        + 0.20 * metrics.get("auprc", 0.0)
        + 0.20 * metrics.get("macro_f1", 0.0)
        + 0.20 * metrics.get("sensitivity_cancer", metrics.get("sensitivity_macro", 0.0))
        + 0.10 * metrics.get("specificity_cancer", metrics.get("specificity_macro", 0.0))
        + 0.05 * (1.0 - metrics.get("ece", 1.0))
    )


def _monitor_value(metrics: dict[str, float], monitor: str) -> float:
    key = monitor.removeprefix("val_")
    value = float(metrics.get(key, -math.inf))
    return value if np.isfinite(value) else -math.inf


def _build_loss(config: dict[str, Any], counts: list[int]) -> Any:
    import torch

    loss_config = ((config.get("imbalance", {}) or {}).get("loss", {}) or {})
    name = str(loss_config.get("name", "cross_entropy")).lower()
    priors = np.asarray(counts, dtype=float) / max(float(sum(counts)), 1.0)
    if name == "balanced_softmax":
        return BalancedSoftmaxLoss(counts)
    if name in {"logit_adjustment", "logit_adjusted_ce"}:
        return LogitAdjustmentLoss(priors.tolist(), tau=float(loss_config.get("tau", 1.0)))
    if name in {"weighted_cross_entropy", "weighted_ce"}:
        weights = torch.as_tensor(
            [sum(counts) / max(count, 1) for count in counts],
            dtype=torch.float32,
        )
        weights = weights / weights.mean().clamp_min(1e-12)
        return torch.nn.CrossEntropyLoss(weight=weights)
    return torch.nn.CrossEntropyLoss()


def _write_logits_frame(
    sample_ids: list[str],
    targets: np.ndarray,
    logits: np.ndarray,
    path: str | Path,
) -> pd.DataFrame:
    frame = pd.DataFrame({"sample_id": sample_ids, "y_true": targets.astype(int)})
    for index in range(logits.shape[1]):
        frame[f"logit_{index}"] = logits[:, index]
    write_table(frame, path)
    return frame


def _collect_logits(model: Any, loader: Any, device: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import torch

    model.eval()
    logits_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device)).detach().cpu().numpy()
            logits_all.append(logits)
            targets_all.append(batch["target"].cpu().numpy())
            sample_ids.extend(batch["sample_id"])
    return np.concatenate(logits_all), np.concatenate(targets_all), sample_ids


def _loss_from_logits(criterion: Any, logits: np.ndarray, targets: np.ndarray, device: str) -> float:
    import torch

    with torch.no_grad():
        logits_tensor = torch.as_tensor(logits, dtype=torch.float32, device=device)
        targets_tensor = torch.as_tensor(targets, dtype=torch.long, device=device)
        loss = criterion(logits_tensor, targets_tensor)
    return float(loss.detach().cpu().item())


def _train_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    device: str,
) -> tuple[float, dict[int, int]]:
    model.train()
    losses: list[float] = []
    sampled: dict[int, int] = {}
    for batch in loader:
        targets = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["image"].to(device))
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        for label in targets.detach().cpu().numpy().astype(int):
            sampled[int(label)] = sampled.get(int(label), 0) + 1
    return float(np.mean(losses)) if losses else math.nan, sampled


def run_v2_experiment(config: dict[str, Any], *, command: str = "") -> dict[str, float]:
    import torch

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    run_dir = ensure_dir((config.get("outputs", {}) or {}).get("run_dir", "outputs/v2/runs/unnamed"))
    task = config.get("task", {}) or {}
    training = config.get("training", {}) or {}
    label_column = str(task.get("label_column", "label_cancer_binary"))
    class_names = list(task.get("classes", [str(index) for index in range(int(task.get("num_classes", 2)))]))
    num_classes = int(task.get("num_classes", len(class_names)))

    train_csv = Path((config.get("data", {}) or {})["train_csv"])
    val_csv = Path((config.get("data", {}) or {})["val_csv"])
    train_frame = pd.read_csv(train_csv)
    val_frame = pd.read_csv(val_csv)
    train_frame = train_frame[train_frame["is_valid"].astype(bool)].reset_index(drop=True)
    val_frame = val_frame[val_frame["is_valid"].astype(bool)].reset_index(drop=True)
    data_config = config.get("data", {}) or {}
    train_frame = _limit_frame_stratified(
        train_frame,
        int(data_config["max_train_samples"]) if data_config.get("max_train_samples") is not None else None,
        label_column=label_column,
        seed=seed,
    )
    val_frame = _limit_frame_stratified(
        val_frame,
        int(data_config["max_val_samples"]) if data_config.get("max_val_samples") is not None else None,
        label_column=label_column,
        seed=seed + 100,
    )
    counts = class_counts(train_frame[label_column].astype(int).tolist(), num_classes)
    priors = [count / max(sum(counts), 1) for count in counts]
    write_json({str(index): count for index, count in enumerate(counts)}, run_dir / "class_counts.json")
    write_json({str(index): prior for index, prior in enumerate(priors)}, run_dir / "class_priors.json")

    sampler = None
    sampler_config = ((config.get("imbalance", {}) or {}).get("sampler", {}) or {})
    if bool(sampler_config.get("enabled", False)):
        sampler = build_class_balanced_sampler(
            train_frame[label_column].astype(int).tolist(),
            num_classes=num_classes,
            mode=str(sampler_config.get("mode", "inverse_frequency")),
            beta=float(sampler_config.get("beta", 0.999)),
        )

    model_config = dict(config.get("model", {}) or {})
    model_config["num_classes"] = num_classes
    device = get_device(prefer_cuda=bool(training.get("prefer_cuda", True)))
    model = create_model(model_config).to(device)
    criterion = _build_loss(config, counts).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.05)),
    )
    train_loader = _build_loader(train_frame, config, train=True, sampler=sampler)
    val_loader = _build_loader(val_frame, config, train=False)

    write_run_manifest(
        run_dir,
        config,
        seed=seed,
        command=command,
        split_paths=[train_csv, val_csv],
    )

    best_score = -math.inf
    best_metrics: dict[str, float] = {}
    patience = int(training.get("patience", 5))
    monitor = str(training.get("monitor", "val_selection_score"))
    epochs_without_improvement = 0
    epoch_rows: list[dict[str, Any]] = []
    sampled_rows: list[dict[str, Any]] = []
    checkpoint_path = run_dir / "checkpoint_best.pt"
    for epoch in range(1, int(training.get("epochs", 1)) + 1):
        train_loss, sampled = _train_epoch(model, train_loader, criterion, optimizer, device)
        logits, targets, sample_ids = _collect_logits(model, val_loader, device)
        probabilities = ensure_probabilities(logits)
        metrics = compute_classification_metrics(targets, y_score=probabilities, class_names=class_names)
        metrics["selection_score"] = _selection_score(metrics)
        metrics["loss"] = _loss_from_logits(criterion, logits, targets, device)
        row = {"epoch": epoch, "train_loss": train_loss}
        row.update({f"val_{key}": value for key, value in metrics.items()})
        epoch_rows.append(row)
        sampled_rows.extend(
            {"epoch": epoch, "class_index": class_index, "sampled_count": count}
            for class_index, count in sorted(sampled.items())
        )
        write_table(pd.DataFrame(epoch_rows), run_dir / "metrics_epoch.csv")
        write_table(pd.DataFrame(sampled_rows), run_dir / "sampled_class_distribution_epoch.csv")
        score = _monitor_value(metrics, monitor)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_metrics = metrics
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": metrics,
                },
                checkpoint_path,
            )
            predictions = build_predictions_frame(sample_ids, targets, probabilities)
            write_table(predictions, run_dir / "predictions_val.csv")
            _write_logits_frame(sample_ids, targets, logits, run_dir / "logits_val.csv")
            write_json(metrics, run_dir / "metrics_val.json")
            by_modality = metrics_by_modality(predictions, val_frame)
            write_table(by_modality, run_dir / "stage04_metrics_by_modality.csv")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if not checkpoint_path.exists():
        save_checkpoint({"model_state": model.state_dict(), "config": config, "epoch": 0}, checkpoint_path)
    return best_metrics
