from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cdg_cysto.augmentations.modality_aware import build_modality_aware_transform  # noqa: E402


def test_v2_modality_aware_transform_reads_sample_modality() -> None:
    transform = build_modality_aware_transform(
        policy="modality_aware_v1",
        train=True,
        input_size=32,
    )
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    result = transform(image=image, sample={"modality": "NBI"})
    assert result["image"].shape == (3, 32, 32)
