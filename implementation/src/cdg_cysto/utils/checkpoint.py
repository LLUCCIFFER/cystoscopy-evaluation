"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        msg = "PyTorch is required to save checkpoints."
        raise ImportError(msg) from exc

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        msg = "PyTorch is required to load checkpoints."
        raise ImportError(msg) from exc

    return torch.load(Path(path), map_location=map_location)  # type: ignore[no-any-return]
