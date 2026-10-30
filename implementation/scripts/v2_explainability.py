"""Grad-CAM explainability evaluation for CDG-Cysto v2."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdg_cysto.explain.cam_metrics import (  # noqa: E402
    cam_iou_top_percent,
    load_cystods_json_mask,
    pointing_game,
    roi_energy,
)
from cdg_cysto.explain.gradcam_vit import save_cam_case_figure  # noqa: E402
from cdg_cysto.models.factory import create_model  # noqa: E402
from cdg_cysto.train.engine_v2 import V2MetadataDataset, _build_transform, collate_v2  # noqa: E402
from cdg_cysto.utils.checkpoint import load_checkpoint  # noqa: E402
from cdg_cysto.utils.device import get_device  # noqa: E402
from cdg_cysto.utils.io import write_table  # noqa: E402


def _reshape_tokens(tensor: Any) -> Any:
    if tensor.ndim != 3:
        return tensor
    batch, tokens, channels = tensor.shape
    height = int(math.sqrt(tokens))
    if height * height != tokens and int(math.sqrt(tokens - 1)) ** 2 == tokens - 1:
        tensor = tensor[:, 1:, :]
        tokens -= 1
        height = int(math.sqrt(tokens))
    width = tokens // height
    return tensor.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()


def _target_layers(model: Any) -> list[Any]:
    if hasattr(model, "layers"):
        layers = model.layers
        last = layers[-1]
        if hasattr(last, "blocks"):
            block = last.blocks[-1]
            return [getattr(block, "norm1", block)]
        return [last]
    if hasattr(model, "stages"):
        stage = model.stages[-1]
        if hasattr(stage, "blocks"):
            return [stage.blocks[-1]]
        return [stage]
    for name in ("conv_head", "layer4"):
        if hasattr(model, name):
            layer = getattr(model, name)
            if name == "layer4" and hasattr(layer, "__getitem__"):
                return [layer[-1]]
            return [layer]
    msg = "Could not infer Grad-CAM target layer for model."
    raise AttributeError(msg)


def _load_model(checkpoint_path: Path) -> tuple[Any, dict[str, Any], str]:
    device = get_device(prefer_cuda=True)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model_config = dict(config.get("model", {}) or {})
    model_config["num_classes"] = int((config.get("task", {}) or {}).get("num_classes", 2))
    model = create_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model, config, device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-csv", default="outputs/v2/splits/cystods_test_internal.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model, config, device = _load_model(Path(args.checkpoint))
    frame = pd.read_csv(args.split_csv)
    frame = frame[frame["has_mask"].astype(bool) & frame["is_valid"].astype(bool)].reset_index(drop=True)
    if args.max_samples is not None:
        frame = frame.head(args.max_samples)
    transform = _build_transform(config, train=False)
    dataset = V2MetadataDataset(frame, label_column="label_cancer_binary", transform=transform)
    cam = GradCAM(model=model, target_layers=_target_layers(model), reshape_transform=_reshape_tokens)
    rows: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    for index in range(len(dataset)):
        item = dataset[index]
        batch = collate_v2([item])
        logits = model(batch["image"].to(device))
        y_pred = int(logits.argmax(dim=1).detach().cpu().item())
        y_true = int(batch["target"].item())
        grayscale = cam(
            input_tensor=batch["image"].to(device),
            targets=[ClassifierOutputTarget(y_pred)],
        )[0]
        row = frame.iloc[index]
        image = np.asarray(Image.open(row["image_path"]).convert("RGB"))
        resized_cam = cv2.resize(grayscale, (image.shape[1], image.shape[0]))
        mask = load_cystods_json_mask(row["mask_path"], width=int(row["width"]), height=int(row["height"]))
        rows.append(
            {
                "sample_id": row["sample_id"],
                "y_true": y_true,
                "y_pred": y_pred,
                "roi_energy": roi_energy(resized_cam, mask),
                "pointing_game": pointing_game(resized_cam, mask),
                "cam_iou_top20": cam_iou_top_percent(resized_cam, mask, top_percent=0.20),
                "cam_iou_top30": cam_iou_top_percent(resized_cam, mask, top_percent=0.30),
            }
        )
        if y_pred == y_true and success_count < 5:
            success_count += 1
            save_cam_case_figure(
                image,
                resized_cam,
                mask,
                output / "success_cases" / f"success_{success_count}_{row['sample_id']}.png",
            )
        if y_pred != y_true and failure_count < 5:
            failure_count += 1
            save_cam_case_figure(
                image,
                resized_cam,
                mask,
                output / "failure_cases" / f"failure_{failure_count}_{row['sample_id']}.png",
            )

    metrics = pd.DataFrame(rows)
    write_table(metrics, output / "cam_metrics.csv")
    summary = metrics[["roi_energy", "pointing_game", "cam_iou_top20", "cam_iou_top30"]].agg(
        ["mean", "std", "count"]
    )
    write_table(summary.reset_index().rename(columns={"index": "stat"}), output / "cam_metrics_summary.csv")
    print(f"Wrote explainability outputs to {output}")


if __name__ == "__main__":
    main()
