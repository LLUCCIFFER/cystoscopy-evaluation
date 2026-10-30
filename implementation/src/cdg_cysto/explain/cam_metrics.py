"""Quantitative CAM metrics using segmentation masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam, dtype=float)
    cam = cam - np.nanmin(cam)
    denom = np.nanmax(cam)
    if denom <= 0:
        return np.zeros_like(cam)
    return cam / denom


def cam_iou(cam: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> float:
    cam_binary = normalize_cam(cam) >= threshold
    mask_binary = np.asarray(mask).astype(bool)
    intersection = np.logical_and(cam_binary, mask_binary).sum()
    union = np.logical_or(cam_binary, mask_binary).sum()
    if union == 0:
        return float("nan")
    return float(intersection / union)


def cam_iou_top_percent(cam: np.ndarray, mask: np.ndarray, top_percent: float = 0.20) -> float:
    """Compute CAM-IoU after thresholding the top percentage of CAM pixels."""

    if not 0 < top_percent <= 1:
        msg = "top_percent must be in (0, 1]."
        raise ValueError(msg)
    normalized = normalize_cam(cam)
    threshold = np.quantile(normalized, 1.0 - top_percent)
    return cam_iou(normalized, mask, threshold=float(threshold))


def roi_energy(cam: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of CAM energy that falls inside the ROI mask."""

    normalized = normalize_cam(cam)
    mask_binary = np.asarray(mask).astype(bool)
    total = float(normalized.sum())
    if total <= 0:
        return float("nan")
    return float(normalized[mask_binary].sum() / total)


def pointing_game(cam: np.ndarray, mask: np.ndarray) -> float:
    normalized = normalize_cam(cam)
    mask_binary = np.asarray(mask).astype(bool)
    max_index = np.unravel_index(int(np.argmax(normalized)), normalized.shape)
    return float(mask_binary[max_index])


def load_cystods_json_mask(json_path: str | Path, width: int, height: int) -> np.ndarray:
    """Load a CystoDS LabelMe-style JSON segmentation as a binary mask."""

    path = Path(json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for shape in payload.get("shapes", []):
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        polygon = [(float(x), float(y)) for x, y in points]
        draw.polygon(polygon, outline=1, fill=1)
    return np.asarray(canvas, dtype=bool)


def summarize_cam_against_mask(cam: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Return the fixed explainability metrics for one CAM/mask pair."""

    return {
        "roi_energy": roi_energy(cam, mask),
        "pointing_game": pointing_game(cam, mask),
        "cam_iou_top20": cam_iou_top_percent(cam, mask, top_percent=0.20),
    }


def deletion_insertion_auc(scores: list[float] | np.ndarray) -> float:
    values = np.asarray(scores, dtype=float)
    if values.size < 2:
        return float("nan")
    x = np.linspace(0.0, 1.0, values.size)
    return float(np.trapz(values, x))
