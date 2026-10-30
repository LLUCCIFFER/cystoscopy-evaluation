"""Grad-CAM helpers for ViT classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch import nn


def build_vit_gradcam(model: nn.Module, target_layers: list[nn.Module]) -> Any:
    try:
        from pytorch_grad_cam import GradCAM
    except ImportError as exc:  # pragma: no cover
        msg = "grad-cam is required for ViT Grad-CAM explanations."
        raise ImportError(msg) from exc

    return GradCAM(model=model, target_layers=target_layers)


def default_dinov2_target_layers(model: nn.Module) -> list[nn.Module]:
    blocks = getattr(getattr(model, "backbone", model), "blocks", None)
    if blocks is None:
        msg = "Cannot find DINOv2 transformer blocks for Grad-CAM target layers."
        raise AttributeError(msg)
    return [blocks[-1].norm1 if hasattr(blocks[-1], "norm1") else blocks[-1]]


def overlay_cam_on_image(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a normalized CAM heatmap on an RGB image."""

    import cv2

    rgb = np.asarray(image, dtype=np.float32)
    if rgb.max() > 1:
        rgb = rgb / 255.0
    heatmap = np.asarray(cam, dtype=np.float32)
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_rgb = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)[:, :, ::-1] / 255.0
    overlay = (1 - alpha) * rgb + alpha * heatmap_rgb
    return np.uint8(np.clip(overlay, 0, 1) * 255)


def save_cam_case_figure(
    image: np.ndarray,
    cam: np.ndarray,
    mask: np.ndarray | None,
    output_path: str | Path,
    *,
    title: str = "",
) -> None:
    """Save an image/CAM/mask panel for success or failure examples."""

    import matplotlib.pyplot as plt

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_cam_on_image(image, cam)
    panels = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, panels, figsize=(4 * panels, 4), dpi=160)
    axes_array = np.atleast_1d(axes)
    axes_array[0].imshow(image)
    axes_array[0].set_title("Image")
    axes_array[1].imshow(overlay)
    axes_array[1].set_title("CAM")
    if mask is not None:
        axes_array[2].imshow(mask, cmap="gray")
        axes_array[2].set_title("Mask")
    for axis in axes_array:
        axis.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)
