"""CDG-Cysto v3 training engine for three-class and lesion-aware experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cdg_cysto.data.label_mapping_v3 import V3_CLASS_NAMES
from cdg_cysto.eval.metrics import (
    build_predictions_frame,
    compute_classification_metrics,
    ensure_probabilities,
)
from cdg_cysto.explain.cam_metrics import (
    load_cystods_json_mask,
    summarize_cam_against_mask,
)
from cdg_cysto.losses.balanced_softmax import BalancedSoftmaxLoss
from cdg_cysto.models.factory import create_model
from cdg_cysto.models.lesion_aware import FeatureAttentionClassifier
from cdg_cysto.samplers.class_balanced_sampler import (
    build_class_balanced_sampler,
    class_counts,
)
from cdg_cysto.utils.checkpoint import save_checkpoint
from cdg_cysto.utils.device import get_device
from cdg_cysto.utils.io import ensure_dir, write_json, write_table
from cdg_cysto.utils.manifest_v2 import write_run_manifest
from cdg_cysto.utils.seed import seed_everything


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


class V3MetadataDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        label_column: str,
        input_size: int,
        train: bool,
        lesion_strategy: str,
        load_masks: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.label_column = label_column
        self.input_size = input_size
        self.train = train
        self.lesion_strategy = lesion_strategy
        self.load_masks = load_masks
        self.targets = self.frame[label_column].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def _load_mask(self, row: pd.Series) -> np.ndarray:
        if not self.load_masks:
            return np.zeros((1, 1), dtype=bool)
        if not _as_bool(row.get("has_mask", False)):
            return np.zeros((1, 1), dtype=bool)
        mask_path = Path(str(row.get("mask_path", "")))
        if not mask_path.exists():
            return np.zeros((1, 1), dtype=bool)
        try:
            return load_cystods_json_mask(mask_path, int(row["width"]), int(row["height"]))
        except (OSError, ValueError):
            return np.zeros((1, 1), dtype=bool)

    def _crop_to_mask(self, image: Any, mask: np.ndarray) -> tuple[Any, np.ndarray]:
        if not self.train or self.lesion_strategy != "mask_guided_crop" or not mask.any():
            return image, mask
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        width, height = image.size
        box_width = max(x1 - x0 + 1, 1)
        box_height = max(y1 - y0 + 1, 1)
        pad = int(0.25 * max(box_width, box_height))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(width, x1 + pad + 1)
        y1 = min(height, y1 + pad + 1)
        cropped = image.crop((x0, y0, x1, y1))
        return cropped, mask[y0:y1, x0:x1]

    def _transform(self, image: Any, mask: np.ndarray) -> tuple[Any, Any]:
        import torch
        from PIL import Image
        from torchvision.transforms import ColorJitter
        from torchvision.transforms import functional as tf

        image, mask = self._crop_to_mask(image, mask)
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        if self.train and torch.rand(()) < 0.5:
            image = tf.hflip(image)
            mask_image = tf.hflip(mask_image)
        if self.train:
            image = ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02)(image)
        image = tf.resize(image, [self.input_size, self.input_size], antialias=True)
        mask_image = tf.resize(mask_image, [self.input_size, self.input_size])
        image_tensor = tf.to_tensor(image)
        image_tensor = tf.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        mask_tensor = torch.from_numpy(np.asarray(mask_image, dtype=np.float32) / 255.0)
        return image_tensor, mask_tensor.unsqueeze(0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        mask = self._load_mask(row)
        image_tensor, mask_tensor = self._transform(image, mask)
        return {
            "image": image_tensor,
            "target": int(row[self.label_column]),
            "mask": mask_tensor,
            "has_mask": bool(mask_tensor.sum().item() > 0),
            "sample_id": str(row["sample_id"]),
            "modality": str(row.get("modality", "")),
            "label_original": str(row.get("label_original", "")),
        }


def collate_v3(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "image": torch.stack([item["image"] for item in batch]),
        "target": torch.as_tensor([item["target"] for item in batch], dtype=torch.long),
        "mask": torch.stack([item["mask"] for item in batch]),
        "has_mask": torch.as_tensor([item["has_mask"] for item in batch], dtype=torch.bool),
        "sample_id": [item["sample_id"] for item in batch],
        "modality": [item["modality"] for item in batch],
        "label_original": [item["label_original"] for item in batch],
    }


def _valid_frame(path: Path, label_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "is_valid" in frame.columns:
        frame = frame[frame["is_valid"].map(_as_bool)].copy()
    frame = frame[frame[label_column].notna()].copy()
    frame[label_column] = frame[label_column].astype(int)
    return frame.reset_index(drop=True)


def _limit_frame_stratified(
    frame: pd.DataFrame,
    max_samples: int | None,
    *,
    label_column: str,
    seed: int,
) -> pd.DataFrame:
    if max_samples is None or len(frame) <= max_samples:
        return frame.reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    sampled_indices: list[int] = []
    labels = list(frame[label_column].value_counts().index)
    per_class = max(1, max_samples // max(len(labels), 1))
    for label in labels:
        subset = frame[frame[label_column] == label]
        sampled = subset.sample(n=min(per_class, len(subset)), random_state=seed + int(label))
        sampled_indices.extend([int(index) for index in sampled.index])
        parts.append(sampled)
    sampled = pd.concat(parts, ignore_index=True)
    remaining = max_samples - len(sampled)
    if remaining > 0:
        rest = frame.drop(sampled_indices, errors="ignore")
        if not rest.empty:
            sampled = pd.concat(
                [sampled, rest.sample(n=min(remaining, len(rest)), random_state=seed + 99)],
                ignore_index=True,
            )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _build_loader(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    train: bool,
    sampler: Any = None,
) -> Any:
    import torch

    model_config = config.get("model", {}) or {}
    training = config.get("training", {}) or {}
    task = config.get("task", {}) or {}
    lesion = config.get("lesion", {}) or {}
    dataset = V3MetadataDataset(
        frame,
        label_column=str(task.get("label_column", "label_v3_three_class")),
        input_size=int(model_config.get("input_size", 224)),
        train=train,
        lesion_strategy=str(lesion.get("strategy", "classification_only")),
        load_masks=bool(lesion.get("enabled", False)),
    )
    num_workers = int(training.get("num_workers", 0))
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(training.get("batch_size", 16)),
        "shuffle": train and sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": collate_v3,
        "pin_memory": bool(training.get("pin_memory", torch.cuda.is_available())),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(training.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(training.get("prefetch_factor", 2))
    return torch.utils.data.DataLoader(
        dataset,
        **loader_kwargs,
    )


def _build_loss(config: dict[str, Any], counts: list[int]) -> Any:
    import torch

    loss_config = ((config.get("imbalance", {}) or {}).get("loss", {}) or {})
    name = str(loss_config.get("name", "cross_entropy")).lower()
    if name == "balanced_softmax":
        return BalancedSoftmaxLoss(counts)
    return torch.nn.CrossEntropyLoss()


def _build_model(config: dict[str, Any], num_classes: int) -> Any:
    model_config = dict(config.get("model", {}) or {})
    lesion = config.get("lesion", {}) or {}
    model_config["num_classes"] = num_classes
    if bool(lesion.get("enabled", False)):
        return FeatureAttentionClassifier(
            name=str(model_config.get("name", "swin_tiny")),
            num_classes=num_classes,
            pretrained=bool(model_config.get("pretrained", True)),
            dropout=float(model_config.get("dropout", 0.0)),
        )
    return create_model(model_config)


def _selection_score(metrics: dict[str, float]) -> float:
    roi_energy = metrics.get("roi_energy", 0.0)
    if not np.isfinite(roi_energy):
        roi_energy = 0.0
    return float(
        0.40 * metrics.get("macro_f1", 0.0)
        + 0.30 * metrics.get("sensitivity_malignant", 0.0)
        + 0.20 * metrics.get("specificity_benign_abnormal", 0.0)
        + 0.10 * float(roi_energy)
    )


def _monitor_value(metrics: dict[str, float], monitor: str) -> float:
    key = monitor.removeprefix("val_")
    value = float(metrics.get(key, -math.inf))
    return value if np.isfinite(value) else -math.inf


def _attention_loss(attention: Any, masks: Any, has_mask: Any) -> Any:
    import torch
    from torch.nn import functional as f

    valid = has_mask.bool()
    if not bool(valid.any()):
        return torch.zeros((), device=attention.device)
    resized_masks = f.interpolate(masks.float(), size=attention.shape[-2:], mode="nearest")
    maps = attention.float()
    flat = maps.flatten(2)
    maps = maps - flat.min(dim=2).values.view(-1, 1, 1, 1)
    denom = maps.flatten(2).max(dim=2).values.view(-1, 1, 1, 1).clamp_min(1e-6)
    maps = maps / denom
    return f.mse_loss(maps[valid], resized_masks[valid])


def _forward_logits(model: Any, images: Any, *, return_attention: bool) -> tuple[Any, Any | None]:
    output = model(images, return_attention=return_attention) if return_attention else model(images)
    if isinstance(output, dict):
        return output["logits"], output.get("attention")
    return output, None


def _train_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    device: str,
    *,
    attention_weight: float,
    use_amp: bool,
    scaler: Any | None,
) -> tuple[float, dict[int, int]]:
    import torch

    model.train()
    losses: list[float] = []
    sampled: dict[int, int] = {}
    use_attention = attention_weight > 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, attention = _forward_logits(model, images, return_attention=use_attention)
            loss = criterion(logits, targets)
            if use_attention and attention is not None:
                loss = loss + attention_weight * _attention_loss(
                    attention,
                    batch["mask"].to(device, non_blocking=True),
                    batch["has_mask"].to(device, non_blocking=True),
                )
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        for label in targets.detach().cpu().numpy().astype(int):
            sampled[int(label)] = sampled.get(int(label), 0) + 1
    return float(np.mean(losses)) if losses else math.nan, sampled


def _collect_outputs(
    model: Any,
    loader: Any,
    device: str,
    *,
    collect_attention: bool,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    import torch
    from torch.nn import functional as f

    model.eval()
    logits_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    sample_ids: list[str] = []
    cam_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, attention = _forward_logits(
                    model,
                    batch["image"].to(device, non_blocking=True),
                    return_attention=collect_attention,
                )
            logits_all.append(logits.detach().float().cpu().numpy())
            targets_all.append(batch["target"].cpu().numpy())
            sample_ids.extend(batch["sample_id"])
            if collect_attention and attention is not None:
                masks = f.interpolate(
                    batch["mask"].float(),
                    size=attention.shape[-2:],
                    mode="nearest",
                ).numpy()
                cams = attention.detach().cpu().numpy()
                for index, sample_id in enumerate(batch["sample_id"]):
                    if masks[index, 0].sum() <= 0:
                        continue
                    row = {
                        "sample_id": sample_id,
                        "modality": batch["modality"][index],
                        "label_original": batch["label_original"][index],
                    }
                    row.update(summarize_cam_against_mask(cams[index, 0], masks[index, 0] > 0))
                    cam_rows.append(row)
    return (
        np.concatenate(logits_all),
        np.concatenate(targets_all),
        sample_ids,
        pd.DataFrame(cam_rows),
    )


def _write_logits_frame(
    sample_ids: list[str],
    targets: np.ndarray,
    logits: np.ndarray,
    path: str | Path,
) -> None:
    frame = pd.DataFrame({"sample_id": sample_ids, "y_true": targets.astype(int)})
    for index in range(logits.shape[1]):
        frame[f"logit_{index}"] = logits[:, index]
    write_table(frame, path)


def _metrics_by_group(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str,
    class_names: list[str],
) -> pd.DataFrame:
    merged = predictions.merge(metadata[["sample_id", group_column]], on="sample_id", how="left")
    rows = []
    prob_columns = [column for column in merged.columns if column.startswith("prob_")]
    for value, group in merged.groupby(group_column, dropna=False):
        metrics = compute_classification_metrics(
            group["y_true"].to_numpy(dtype=int),
            y_pred=group["y_pred"].to_numpy(dtype=int),
            y_score=group[prob_columns].to_numpy(dtype=float),
            class_names=class_names,
        )
        rows.append({group_column: value, "n": len(group), **metrics})
    return pd.DataFrame(rows)


def run_v3_experiment(config: dict[str, Any], *, command: str = "") -> dict[str, float]:
    import torch

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    run_dir = ensure_dir((config.get("outputs", {}) or {}).get("run_dir", "outputs/v3/runs/unnamed"))
    task = config.get("task", {}) or {}
    training = config.get("training", {}) or {}
    label_column = str(task.get("label_column", "label_v3_three_class"))
    class_names = list(task.get("classes", V3_CLASS_NAMES))
    num_classes = int(task.get("num_classes", len(class_names)))
    train_csv = Path((config.get("data", {}) or {})["train_csv"])
    val_csv = Path((config.get("data", {}) or {})["val_csv"])
    train_frame = _valid_frame(train_csv, label_column)
    val_frame = _valid_frame(val_csv, label_column)
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
    write_json({str(index): count for index, count in enumerate(counts)}, run_dir / "class_counts.json")

    sampler = None
    sampler_config = ((config.get("imbalance", {}) or {}).get("sampler", {}) or {})
    if bool(sampler_config.get("enabled", False)):
        sampler = build_class_balanced_sampler(
            train_frame[label_column].astype(int).tolist(),
            num_classes=num_classes,
            mode=str(sampler_config.get("mode", "inverse_frequency")),
            beta=float(sampler_config.get("beta", 0.999)),
        )

    device = get_device(prefer_cuda=bool(training.get("prefer_cuda", True)))
    use_amp = bool(training.get("amp", True)) and str(device).startswith("cuda")
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = bool(training.get("cudnn_benchmark", True))
        torch.set_float32_matmul_precision(str(training.get("matmul_precision", "high")))
    model = _build_model(config, num_classes).to(device)
    criterion = _build_loss(config, counts).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", training.get("lr", 1e-4))),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_loader = _build_loader(train_frame, config, train=True, sampler=sampler)
    val_loader = _build_loader(val_frame, config, train=False)
    write_run_manifest(run_dir, config, seed=seed, command=command, split_paths=[train_csv, val_csv])

    best_score = -math.inf
    best_metrics: dict[str, float] = {}
    patience = int(training.get("patience", training.get("early_stopping_patience", 5)))
    monitor = str(training.get("monitor", "val_macro_f1"))
    epochs_without_improvement = 0
    epoch_rows: list[dict[str, Any]] = []
    sampled_rows: list[dict[str, Any]] = []
    checkpoint_path = run_dir / "checkpoint_best.pt"
    lesion = config.get("lesion", {}) or {}
    attention_weight = float(lesion.get("attention_weight", 0.0))
    collect_attention = bool(lesion.get("enabled", False))

    for epoch in range(1, int(training.get("epochs", 1)) + 1):
        train_loss, sampled = _train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            attention_weight=attention_weight,
            use_amp=use_amp,
            scaler=scaler,
        )
        logits, targets, sample_ids, cam_metrics = _collect_outputs(
            model,
            val_loader,
            device,
            collect_attention=collect_attention,
            use_amp=use_amp,
        )
        probabilities = ensure_probabilities(logits)
        metrics = compute_classification_metrics(
            targets,
            y_score=probabilities,
            class_names=class_names,
        )
        if not cam_metrics.empty:
            metrics.update(
                {
                    "roi_energy": float(cam_metrics["roi_energy"].mean()),
                    "pointing_game": float(cam_metrics["pointing_game"].mean()),
                    "cam_iou_top20": float(cam_metrics["cam_iou_top20"].mean()),
                }
            )
        metrics["selection_score"] = _selection_score(metrics)
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
            write_table(cam_metrics, run_dir / "cam_metrics_val.csv")
            write_table(
                _metrics_by_group(predictions, val_frame, "modality", class_names),
                run_dir / "metrics_by_modality_val.csv",
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if not checkpoint_path.exists():
        save_checkpoint({"model_state": model.state_dict(), "config": config, "epoch": 0}, checkpoint_path)
    return best_metrics
