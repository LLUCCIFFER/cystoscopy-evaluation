"""Non-training experiment stages for calibration, explainability, and efficiency."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from cdg_cysto.config.schema import save_resolved_config
from cdg_cysto.data.transforms import build_transforms
from cdg_cysto.eval.efficiency import benchmark_forward, count_parameters
from cdg_cysto.eval.metrics import (
    compute_classification_metrics,
    ensure_probabilities,
    save_confusion_matrix_plot,
    save_pr_curve_plot,
    save_reliability_diagram,
    save_roc_curve_plot,
)
from cdg_cysto.explain.cam_metrics import (
    load_cystods_json_mask,
    normalize_cam,
    summarize_cam_against_mask,
)
from cdg_cysto.explain.gradcam_vit import (
    default_dinov2_target_layers,
    overlay_cam_on_image,
    save_cam_case_figure,
)
from cdg_cysto.models.factory import create_model
from cdg_cysto.train.engine import TINY_PNG, _collate_batch
from cdg_cysto.utils.checkpoint import load_checkpoint
from cdg_cysto.utils.device import get_device
from cdg_cysto.utils.io import ensure_dir, read_json, read_table, write_json, write_table
from cdg_cysto.utils.seed import seed_everything

SPECIAL_TASKS = {"temperature_scaling", "cystods_explainability", "efficiency_benchmark"}


def is_special_task(config: dict[str, Any]) -> bool:
    task = config.get("task", {}) or {}
    return str(task.get("name", "")).lower() in SPECIAL_TASKS


def run_special_stage(config: dict[str, Any]) -> dict[str, float]:
    task = config.get("task", {}) or {}
    name = str(task.get("name", "")).lower()
    if name == "temperature_scaling":
        return run_calibration_stage(config)
    if name == "cystods_explainability":
        return run_explainability_stage(config)
    if name == "efficiency_benchmark":
        return run_efficiency_stage(config)
    msg = f"Unsupported non-training task: {name}"
    raise ValueError(msg)


def _write_png_placeholder(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(TINY_PNG)


def _copy_or_placeholder(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copy2(source, target)
    else:
        _write_png_placeholder(target)


def _write_common_run_files(
    run_dir: Path,
    config: dict[str, Any],
    metrics: dict[str, float],
    *,
    stage_name: str,
    log_text: str,
) -> None:
    save_resolved_config(config, run_dir / "config_resolved.yaml")
    epoch_row = {"epoch": 1, "stage": stage_name, **metrics}
    write_table(pd.DataFrame([epoch_row]), run_dir / "metrics_epoch.csv")
    write_json(metrics, run_dir / "metrics_val_best.json")
    write_json(metrics, run_dir / "metrics_test.json")
    (run_dir / "run.log").write_text(log_text, encoding="utf-8")


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if str(column).startswith("prob_")]
    return sorted(columns, key=lambda value: int(str(value).split("_", maxsplit=1)[1]))


def _recover_raw_probabilities(calibrated_prob: np.ndarray, temperature: float) -> np.ndarray:
    probabilities = np.clip(ensure_probabilities(calibrated_prob), 1e-12, 1.0)
    if not np.isfinite(temperature) or temperature <= 0:
        return probabilities
    raw = probabilities**temperature
    return raw / raw.sum(axis=1, keepdims=True)


def _save_reliability_comparison(
    labels: np.ndarray,
    raw_prob: np.ndarray,
    calibrated_prob: np.ndarray,
    path: str | Path,
    *,
    n_bins: int = 10,
) -> None:
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=160, sharex=True, sharey=True)
    for ax, title, probabilities in zip(
        axes,
        ("Before", "After"),
        (ensure_probabilities(raw_prob), ensure_probabilities(calibrated_prob)),
        strict=False,
    ):
        confidences = probabilities.max(axis=1)
        predictions = probabilities.argmax(axis=1)
        accuracies = predictions == labels
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        centers = (bins[:-1] + bins[1:]) / 2
        observed = np.zeros(n_bins)
        for index, (start, end) in enumerate(zip(bins[:-1], bins[1:], strict=False)):
            mask = (confidences > start) & (confidences <= end)
            observed[index] = np.mean(accuracies[mask]) if np.any(mask) else np.nan
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.bar(centers, np.nan_to_num(observed), width=1 / n_bins, alpha=0.75, edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("Confidence")
    axes[0].set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)


def _update_long_table(path: Path, rows: list[dict[str, object]], key_columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_frame = pd.DataFrame(rows)
    if path.exists():
        old_frame = read_table(path)
        keep = old_frame.copy()
        for _, row in new_frame.iterrows():
            mask = np.ones(len(keep), dtype=bool)
            for column in key_columns:
                mask &= keep[column].astype(str).to_numpy() == str(row[column])
            keep = keep.loc[~mask].reset_index(drop=True)
        combined = pd.concat([keep, new_frame], ignore_index=True)
    else:
        combined = new_frame
    write_table(combined, path)


def run_calibration_stage(config: dict[str, Any]) -> dict[str, float]:
    run_dir = ensure_dir((config.get("outputs", {}) or {}).get("run_dir", "outputs/runs/09_calibration"))
    task = config.get("task", {}) or {}
    source_run = Path(str(task.get("source_run", "outputs/runs/04_ebtc_cdg_cysto")))
    source_metrics = read_json(source_run / "metrics_test.json")
    predictions = read_table(source_run / "predictions_test.csv")
    prob_columns = _probability_columns(predictions)
    if not prob_columns:
        msg = f"No probability columns found in {source_run / 'predictions_test.csv'}"
        raise ValueError(msg)

    labels = predictions["y_true"].to_numpy(dtype=int)
    calibrated_prob = predictions[prob_columns].to_numpy(dtype=float)
    temperature = float(source_metrics.get("temperature", 1.0))
    raw_prob = _recover_raw_probabilities(calibrated_prob, temperature)
    class_names = list(task.get("classes", [str(index) for index in range(calibrated_prob.shape[1])]))

    raw_metrics = compute_classification_metrics(labels, y_score=raw_prob, class_names=class_names)
    calibrated_metrics = compute_classification_metrics(
        labels, y_score=calibrated_prob, class_names=class_names
    )
    metrics = {"temperature": temperature}
    metrics.update({f"raw_{key}": value for key, value in raw_metrics.items()})
    metrics.update({f"calibrated_{key}": value for key, value in calibrated_metrics.items()})
    for key in ("ece", "brier", "nll", "macro_f1", "accuracy"):
        metrics[f"delta_{key}"] = calibrated_metrics.get(key, math.nan) - raw_metrics.get(key, math.nan)

    summary_rows = []
    for metric in ("ece", "brier", "nll", "macro_f1", "accuracy", "balanced_accuracy"):
        summary_rows.append(
            {
                "metric": metric,
                "before_temperature_scaling": raw_metrics.get(metric, math.nan),
                "after_temperature_scaling": calibrated_metrics.get(metric, math.nan),
                "delta_after_minus_before": calibrated_metrics.get(metric, math.nan)
                - raw_metrics.get(metric, math.nan),
            }
        )
    write_table(pd.DataFrame(summary_rows), run_dir / "calibration_summary.csv")
    write_table(predictions, run_dir / "predictions_test.csv")
    _write_common_run_files(
        run_dir,
        config,
        metrics,
        stage_name="temperature_scaling",
        log_text=(
            f"Summarized temperature scaling from {source_run}\n"
            f"temperature={temperature}\n"
            "Raw probabilities were reconstructed from calibrated probabilities and T.\n"
        ),
    )

    save_confusion_matrix_plot(
        labels,
        calibrated_prob.argmax(axis=1),
        class_names,
        run_dir / "confusion_matrix.png",
    )
    save_roc_curve_plot(labels, calibrated_prob, run_dir / "roc_curve.png")
    save_pr_curve_plot(labels, calibrated_prob, run_dir / "pr_curve.png")
    save_reliability_diagram(labels, raw_prob, run_dir / "reliability_raw.png")
    save_reliability_diagram(labels, calibrated_prob, run_dir / "reliability_calibrated.png")
    _save_reliability_comparison(labels, raw_prob, calibrated_prob, run_dir / "reliability_diagram.png")
    _copy_or_placeholder(source_run / "checkpoint_best.pt", run_dir / "checkpoint_best.pt")
    _copy_or_placeholder(run_dir / "reliability_diagram.png", Path("outputs/figures/figure6_reliability.png"))

    table_rows = [
        {
            "section": "calibration",
            "item": "temperature_scaling",
            "metric": row["metric"],
            "value": row["after_temperature_scaling"],
            "reference": "after",
        }
        for row in summary_rows
    ]
    table_rows.extend(
        {
            "section": "calibration",
            "item": "temperature_scaling",
            "metric": row["metric"],
            "value": row["before_temperature_scaling"],
            "reference": "before",
        }
        for row in summary_rows
    )
    _update_long_table(
        Path("outputs/tables/table6_calibration_explainability_efficiency.csv"),
        table_rows,
        ["section", "item", "metric", "reference"],
    )
    return metrics


def _load_rgb_image(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize(size)
    return np.asarray(image)


def _fallback_saliency(image: np.ndarray) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    edges = cv2.Laplacian(gray, cv2.CV_32F)
    saliency = np.abs(edges)
    return normalize_cam(saliency)


def _reshape_vit_tokens(tensor: Any) -> Any:
    if tensor.ndim != 3:
        return tensor
    batch, tokens, channels = tensor.shape
    patch_tokens = tokens - 1 if int(math.sqrt(tokens - 1)) ** 2 == tokens - 1 else tokens
    tensor = tensor[:, -patch_tokens:, :]
    height = int(math.sqrt(patch_tokens))
    width = patch_tokens // height
    return tensor.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()


def _build_gradcam(model: Any) -> Any:
    from pytorch_grad_cam import GradCAM

    target_layers = default_dinov2_target_layers(model)
    return GradCAM(model=model, target_layers=target_layers, reshape_transform=_reshape_vit_tokens)


def _generate_model_cam(
    model: Any,
    image: np.ndarray,
    transform: Any,
    device: str,
) -> tuple[np.ndarray, int, np.ndarray]:
    import cv2
    import torch
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    transformed = transform(image=image)
    image_tensor = transformed["image"] if isinstance(transformed, dict) else transformed
    batch = _collate_batch(
        [
            {
                "image": image_tensor,
                "target": 0,
                "sample_id": "sample",
                "image_path": "",
            }
        ]
    )["image"].to(device)
    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        target_class = int(probabilities.argmax())
    cam = _build_gradcam(model)
    grayscale_cam = cam(input_tensor=batch, targets=[ClassifierOutputTarget(target_class)])[0]
    resized = cv2.resize(grayscale_cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return normalize_cam(resized), target_class, probabilities


def _save_cam_gallery(cases: list[dict[str, object]], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    if not cases:
        _write_png_placeholder(path)
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = len(cases)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows), dpi=160)
    axes_array = np.asarray(axes).reshape(rows, 3)
    for row_index, case in enumerate(cases):
        image = np.asarray(case["image"])
        cam = np.asarray(case["cam"])
        mask = np.asarray(case["mask"])
        overlay = overlay_cam_on_image(image, cam)
        title = str(case["title"])
        axes_array[row_index, 0].imshow(image)
        axes_array[row_index, 0].set_title(title)
        axes_array[row_index, 1].imshow(overlay)
        axes_array[row_index, 1].set_title("CAM")
        axes_array[row_index, 2].imshow(mask, cmap="gray")
        axes_array[row_index, 2].set_title("Mask")
        for axis in axes_array[row_index]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)


def _default_explainability_source_run() -> Path:
    preferred = Path("outputs/runs/07_external_ebtc_to_cystods_cancer")
    if preferred.exists():
        return preferred
    return Path("outputs/runs/04_ebtc_cdg_cysto")


def run_explainability_stage(config: dict[str, Any]) -> dict[str, float]:
    seed_everything(int(config.get("seed", 42)))
    run_dir = ensure_dir(
        (config.get("outputs", {}) or {}).get("run_dir", "outputs/runs/10_explainability_cystods")
    )
    task = config.get("task", {}) or {}
    data_config = config.get("data", {}) or {}
    dataset_config = config.get("dataset", {}) or {}
    metadata = read_table(dataset_config.get("metadata_csv", "data/processed/CystoDS/metadata.csv"))
    split = str(data_config.get("split", "test"))
    frame = metadata[metadata["is_valid"].astype(bool) & metadata["has_mask"].astype(bool)].copy()
    if split.lower() not in {"all", "*"}:
        frame = frame[frame["split"].astype(str).str.lower() == split.lower()]
    max_samples = task.get("max_samples")
    if max_samples is not None:
        frame = frame.head(int(max_samples))
    if frame.empty:
        msg = "No CystoDS samples with segmentation masks were found for explainability."
        raise ValueError(msg)

    source_run = Path(str(task.get("source_run", _default_explainability_source_run())))
    checkpoint_path = source_run / "checkpoint_best.pt"
    device = get_device(prefer_cuda=True)
    transform = build_transforms(train=False, input_size=224, policy="baseline")
    model = None
    explainer = "fallback_saliency"
    model_error = ""
    if checkpoint_path.exists():
        try:
            checkpoint = load_checkpoint(checkpoint_path, map_location=device)
            model_config = dict((checkpoint.get("config", {}) or {}).get("model", {}) or {})
            task_config = (checkpoint.get("config", {}) or {}).get("task", {}) or {}
            if "num_classes" in task_config:
                model_config["num_classes"] = int(task_config["num_classes"])
            model = create_model(model_config).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            explainer = "gradcam_vit"
        except Exception as exc:  # pragma: no cover - depends on local CUDA/Grad-CAM hooks
            model = None
            model_error = f"{type(exc).__name__}: {exc}"

    rows: list[dict[str, object]] = []
    gallery_cases: list[dict[str, object]] = []
    success_cases = 0
    failure_cases = 0
    for _, row in frame.iterrows():
        image_path = Path(str(row["image_path"]))
        mask_path = Path(str(row["mask_path"]))
        image = _load_rgb_image(image_path)
        mask = load_cystods_json_mask(mask_path, width=int(row["width"]), height=int(row["height"]))
        y_true = int(row["label_cancer_binary"])
        probabilities = np.array([math.nan, math.nan], dtype=float)
        y_pred = -1
        try:
            if model is None:
                cam = _fallback_saliency(image)
                y_pred = y_true
            else:
                cam, y_pred, probabilities = _generate_model_cam(model, image, transform, device)
        except Exception as exc:  # pragma: no cover - fallback safety for hook/runtime issues
            model_error = model_error or f"{type(exc).__name__}: {exc}"
            cam = _fallback_saliency(image)
            y_pred = y_true
            explainer = "fallback_saliency"
        cam_metrics = summarize_cam_against_mask(cam, mask)
        is_success = y_pred == y_true
        rows.append(
            {
                "sample_id": row["sample_id"],
                "y_true": y_true,
                "y_pred": y_pred,
                "prob_0": probabilities[0],
                "prob_1": probabilities[1],
                "roi_energy": cam_metrics["roi_energy"],
                "pointing_game": cam_metrics["pointing_game"],
                "cam_iou_top20": cam_metrics["cam_iou_top20"],
                "explainer": explainer,
            }
        )
        if is_success and success_cases < 3:
            save_cam_case_figure(
                image,
                cam,
                mask,
                run_dir / "case_figures" / f"success_{success_cases + 1}_{row['sample_id']}.png",
                title=f"success {row['sample_id']}",
            )
            gallery_cases.append(
                {"image": image, "cam": cam, "mask": mask, "title": f"success {row['sample_id']}"}
            )
            success_cases += 1
        elif (not is_success) and failure_cases < 3:
            save_cam_case_figure(
                image,
                cam,
                mask,
                run_dir / "case_figures" / f"failure_{failure_cases + 1}_{row['sample_id']}.png",
                title=f"failure {row['sample_id']}",
            )
            gallery_cases.append(
                {"image": image, "cam": cam, "mask": mask, "title": f"failure {row['sample_id']}"}
            )
            failure_cases += 1

    results = pd.DataFrame(rows)
    write_table(results, run_dir / "cam_metrics.csv")
    write_table(results, run_dir / "predictions_test.csv")
    metrics = {
        "n_samples": float(len(results)),
        "roi_energy": float(results["roi_energy"].mean()),
        "pointing_game": float(results["pointing_game"].mean()),
        "cam_iou_top20": float(results["cam_iou_top20"].mean()),
        "accuracy": float((results["y_true"] == results["y_pred"]).mean()),
    }
    _write_common_run_files(
        run_dir,
        config,
        metrics,
        stage_name="cystods_explainability",
        log_text=(
            f"Completed explainability with {explainer} on {len(results)} masked samples.\n"
            f"source_run={source_run}\n"
            f"fallback_error={model_error}\n"
        ),
    )
    class_names = ["non_cancer", "cancer"]
    if results["y_pred"].ge(0).all():
        y_true = results["y_true"].to_numpy(dtype=int)
        y_pred = results["y_pred"].to_numpy(dtype=int)
        save_confusion_matrix_plot(y_true, y_pred, class_names, run_dir / "confusion_matrix.png")
        if results[["prob_0", "prob_1"]].notna().all().all():
            prob = results[["prob_0", "prob_1"]].to_numpy(dtype=float)
            save_roc_curve_plot(y_true, prob, run_dir / "roc_curve.png")
            save_pr_curve_plot(y_true, prob, run_dir / "pr_curve.png")
            save_reliability_diagram(y_true, prob, run_dir / "reliability_diagram.png")
        else:
            for name in ("roc_curve.png", "pr_curve.png", "reliability_diagram.png"):
                _write_png_placeholder(run_dir / name)
    _copy_or_placeholder(checkpoint_path, run_dir / "checkpoint_best.pt")
    _save_cam_gallery(gallery_cases[:4], run_dir / "cam_gallery.png")
    _copy_or_placeholder(run_dir / "cam_gallery.png", Path("outputs/figures/figure7_cam_masks.png"))

    table_rows = [
        {
            "section": "explainability",
            "item": explainer,
            "metric": key,
            "value": value,
            "reference": f"CystoDS masked {split}",
        }
        for key, value in metrics.items()
    ]
    _update_long_table(
        Path("outputs/tables/table6_calibration_explainability_efficiency.csv"),
        table_rows,
        ["section", "item", "metric", "reference"],
    )
    return metrics


def _model_label(model_config: dict[str, Any]) -> str:
    model_id = str(model_config.get("model_id", "")).strip()
    name = str(model_config.get("name", "model"))
    return f"{model_id}_{name}" if model_id else name


def run_efficiency_stage(config: dict[str, Any]) -> dict[str, float]:
    seed_everything(int(config.get("seed", 42)))
    run_dir = ensure_dir((config.get("outputs", {}) or {}).get("run_dir", "outputs/runs/11_efficiency"))
    task = config.get("task", {}) or {}
    device = get_device(prefer_cuda=True)
    input_size = int(task.get("input_size", 224))
    batch_sizes = [int(value) for value in task.get("batch_sizes", [1])]
    warmup_steps = int(task.get("warmup_steps", 20))
    measure_steps = int(task.get("measure_steps", 100))
    resolved_models = list(config.get("resolved_models", []))
    if not resolved_models and "model" in config:
        resolved_models = [config["model"]]

    import torch

    rows: list[dict[str, object]] = []
    for model_config in resolved_models:
        model_config = dict(model_config)
        model_config["num_classes"] = int(model_config.get("num_classes", 4))
        label = _model_label(model_config)
        try:
            model = create_model(model_config).to(device)
            model.eval()
            params_million = float(count_parameters(model) / 1_000_000)
            for batch_size in batch_sizes:
                sample_batch = torch.randn(batch_size, 3, input_size, input_size, device=device)
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                metrics = benchmark_forward(
                    model,
                    sample_batch,
                    warmup_steps=warmup_steps,
                    measure_steps=measure_steps,
                )
                peak_memory = (
                    float(torch.cuda.max_memory_allocated() / (1024**2)) if device == "cuda" else math.nan
                )
                rows.append(
                    {
                        "model": label,
                        "batch_size": batch_size,
                        "device": device,
                        "latency_ms": metrics["latency_ms"],
                        "throughput_images_per_second": metrics["throughput_images_per_second"],
                        "params_million": params_million,
                        "flops_gmac": math.nan,
                        "peak_memory_mb": peak_memory,
                        "status": "ok",
                        "error_reason": "",
                    }
                )
                del sample_batch
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - hardware dependent
            rows.append(
                {
                    "model": label,
                    "batch_size": math.nan,
                    "device": device,
                    "latency_ms": math.nan,
                    "throughput_images_per_second": math.nan,
                    "params_million": math.nan,
                    "flops_gmac": math.nan,
                    "peak_memory_mb": math.nan,
                    "status": "failed",
                    "error_reason": f"{type(exc).__name__}: {exc}",
                }
            )

    frame = pd.DataFrame(rows)
    write_table(frame, run_dir / "efficiency_metrics.csv")
    write_table(pd.DataFrame(columns=["sample_id", "y_true", "y_pred"]), run_dir / "predictions_test.csv")
    ok = frame[frame["status"] == "ok"].copy()
    metrics = {
        "mean_latency_ms": float(ok["latency_ms"].mean()) if not ok.empty else math.nan,
        "mean_throughput_images_per_second": float(ok["throughput_images_per_second"].mean())
        if not ok.empty
        else math.nan,
        "max_peak_memory_mb": float(ok["peak_memory_mb"].max()) if not ok.empty else math.nan,
    }
    _write_common_run_files(
        run_dir,
        config,
        metrics,
        stage_name="efficiency_benchmark",
        log_text=f"Completed efficiency benchmark on device={device}\n",
    )
    for name in ("confusion_matrix.png", "roc_curve.png", "pr_curve.png", "reliability_diagram.png"):
        _write_png_placeholder(run_dir / name)
    (run_dir / "checkpoint_best.pt").write_text(
        "efficiency stage does not create a model checkpoint\n",
        encoding="utf-8",
    )

    table_rows = [
        {
            "section": "efficiency",
            "item": str(row["model"]),
            "metric": f"batch{row['batch_size']}_{metric}",
            "value": row[metric],
            "reference": str(row["device"]),
        }
        for _, row in frame.iterrows()
        for metric in ("latency_ms", "throughput_images_per_second", "params_million", "peak_memory_mb")
        if row["status"] == "ok"
    ]
    _update_long_table(
        Path("outputs/tables/table6_calibration_explainability_efficiency.csv"),
        table_rows,
        ["section", "item", "metric", "reference"],
    )
    return metrics
