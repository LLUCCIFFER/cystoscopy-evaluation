"""Efficiency measurement helpers."""

from __future__ import annotations

import time

import torch
from torch import nn


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def benchmark_forward(
    model: nn.Module,
    sample_batch: torch.Tensor,
    *,
    warmup_steps: int = 20,
    measure_steps: int = 100,
) -> dict[str, float]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        msg = "PyTorch is required for efficiency benchmarking."
        raise ImportError(msg) from exc

    model.eval()
    with torch.no_grad():
        for _ in range(warmup_steps):
            model(sample_batch)
        if sample_batch.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(measure_steps):
            model(sample_batch)
        if sample_batch.is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    images = int(sample_batch.shape[0] * measure_steps)
    return {
        "latency_ms": float(elapsed / measure_steps * 1000.0),
        "throughput_images_per_second": float(images / elapsed),
        "params_million": float(count_parameters(model) / 1_000_000),
    }
