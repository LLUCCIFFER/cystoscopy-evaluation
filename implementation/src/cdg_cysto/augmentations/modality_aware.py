"""Modality-aware augmentation policies for CDG-Cysto v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModalityStats:
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


class ModalityAwareTransform:
    """Torchvision-backed transform that can inspect sample modality."""

    uses_sample = True

    def __init__(
        self,
        *,
        policy: str = "basic",
        train: bool = True,
        input_size: int = 224,
        modality_stats: dict[str, ModalityStats] | None = None,
    ) -> None:
        self.policy = policy
        self.train = train
        self.input_size = input_size
        self.modality_stats = modality_stats or {}

    def _ops(self, modality: str):
        from torchvision import transforms

        ops: list[Any] = [transforms.Resize((self.input_size, self.input_size))]
        if self.train:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
            if self.policy in {"modality_aware_v1", "modality_aware_v2_stronger"}:
                if modality.upper() in {"NBI"}:
                    ops.append(transforms.ColorJitter(brightness=0.10, contrast=0.25, saturation=0.10))
                else:
                    ops.append(transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15))
                ops.append(transforms.RandomRotation(degrees=10))
            if self.policy == "modality_aware_v2_stronger":
                ops.extend(
                    [
                        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
                        transforms.RandomGrayscale(p=0.05),
                    ]
                )
        stats = self.modality_stats.get(modality.upper(), ModalityStats())
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=stats.mean, std=stats.std),
            ]
        )
        return transforms.Compose(ops)

    def __call__(self, *, image: Any, sample: dict[str, Any] | None = None) -> dict[str, Any]:
        from PIL import Image

        modality = str((sample or {}).get("modality", "default")).upper()
        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
        return {"image": self._ops(modality)(pil_image)}


def build_modality_aware_transform(
    *,
    policy: str,
    train: bool,
    input_size: int = 224,
    modality_stats: dict[str, ModalityStats] | None = None,
) -> ModalityAwareTransform:
    if policy not in {"basic", "modality_aware_v1", "modality_aware_v2_stronger"}:
        msg = f"Unsupported modality-aware augmentation policy: {policy}"
        raise ValueError(msg)
    return ModalityAwareTransform(
        policy=policy,
        train=train,
        input_size=input_size,
        modality_stats=modality_stats,
    )


def estimate_modality_stats(
    metadata: pd.DataFrame,
    *,
    image_column: str = "image_path",
    modality_column: str = "modality",
    max_images_per_modality: int = 128,
) -> dict[str, ModalityStats]:
    from PIL import Image

    stats: dict[str, ModalityStats] = {}
    for modality, group in metadata.groupby(modality_column):
        pixels: list[np.ndarray] = []
        for path in group[image_column].head(max_images_per_modality):
            try:
                image = Image.open(path).convert("RGB").resize((224, 224))
            except OSError:
                continue
            pixels.append(np.asarray(image, dtype=np.float32).reshape(-1, 3) / 255.0)
        if not pixels:
            continue
        values = np.concatenate(pixels, axis=0)
        mean = tuple(float(value) for value in values.mean(axis=0))
        std = tuple(float(max(value, 1e-6)) for value in values.std(axis=0))
        stats[str(modality).upper()] = ModalityStats(mean=mean, std=std)  # type: ignore[arg-type]
    return stats


def metrics_by_modality(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    sample_column: str = "sample_id",
) -> pd.DataFrame:
    from cdg_cysto.eval.metrics import compute_classification_metrics

    merged = predictions.merge(
        metadata[[sample_column, "modality"]],
        on=sample_column,
        how="left",
    )
    prob_columns = [column for column in merged.columns if column.startswith("prob_")]
    rows: list[dict[str, float | str | int]] = []
    for modality, group in merged.groupby("modality", dropna=False):
        y_true = group["y_true"].to_numpy(dtype=int)
        y_pred = group["y_pred"].to_numpy(dtype=int)
        y_score = group[prob_columns].to_numpy(dtype=float) if prob_columns else None
        metrics = compute_classification_metrics(
            y_true,
            y_pred=y_pred,
            y_score=y_score,
            class_names=["non_cancer", "cancer"] if len(prob_columns) == 2 else None,
        )
        rows.append({"modality": str(modality), "n": len(group), **metrics})
    return pd.DataFrame(rows)
