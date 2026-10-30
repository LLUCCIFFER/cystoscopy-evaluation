"""Device selection helpers."""

from __future__ import annotations


def get_device(prefer_cuda: bool = True) -> str:
    try:
        import torch

        if prefer_cuda and torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
