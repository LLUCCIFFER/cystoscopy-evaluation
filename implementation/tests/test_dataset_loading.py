from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from cdg_cysto.data.dataset import MetadataImageDataset


def test_metadata_image_dataset_loads_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_path)
    frame = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "image_path": [image_path.name],
            "label_ebtc4": [0],
            "pid": ["p1"],
        }
    )
    dataset = MetadataImageDataset(frame, classes=["HGC", "LGC"], image_root=tmp_path)
    sample = dataset[0]
    assert sample["target"] == 0
    assert sample["patient_id"] == "p1"
    assert sample["sample_id"] == "s1"
    assert sample["image"].size == (8, 8)
