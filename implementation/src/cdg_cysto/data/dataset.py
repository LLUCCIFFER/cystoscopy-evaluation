"""PyTorch dataset backed by prepared metadata CSV files."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from cdg_cysto.data.label_mapping import encode_labels
from cdg_cysto.utils.io import read_table


class MetadataImageDataset:
    """Image classification dataset using a metadata table.

    The table must contain at least image, label, and patient columns. Image
    paths may be absolute or relative to `image_root`.
    """

    def __init__(
        self,
        metadata: pd.DataFrame | str | Path,
        classes: list[str] | tuple[str, ...],
        image_root: str | Path | None = None,
        transform: Callable[..., Any] | None = None,
        image_column: str = "image_path",
        label_column: str = "label_ebtc4",
        patient_column: str = "pid",
        sample_column: str = "sample_id",
    ) -> None:
        self.frame = read_table(metadata) if isinstance(metadata, str | Path) else metadata.copy()
        self.classes = list(classes)
        self.image_root = Path(image_root) if image_root is not None else None
        self.transform = transform
        self.image_column = image_column
        if label_column not in self.frame.columns and "label" in self.frame.columns:
            label_column = "label"
        if patient_column not in self.frame.columns and "patient_id" in self.frame.columns:
            patient_column = "patient_id"
        self.label_column = label_column
        self.patient_column = patient_column
        self.sample_column = sample_column

        missing = {image_column, label_column, patient_column} - set(self.frame.columns)
        if missing:
            msg = f"Metadata is missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        if sample_column not in self.frame.columns:
            self.frame[sample_column] = [f"sample_{index:06d}" for index in range(len(self.frame))]

        labels = self.frame[label_column]
        if pd.api.types.is_numeric_dtype(labels):
            self.targets = [int(value) for value in labels.tolist()]
        else:
            self.targets = encode_labels(labels.tolist(), self.classes)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        path = Path(str(row[self.image_column]))
        if not path.is_absolute() and self.image_root is not None:
            path = self.image_root / path

        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            transformed = self.transform(image=np.asarray(image))
            image = transformed["image"] if isinstance(transformed, dict) else transformed

        return {
            "image": image,
            "target": self.targets[index],
            "label": int(self.targets[index]),
            "patient_id": str(row[self.patient_column]),
            "pid": str(row[self.patient_column]),
            "sample_id": str(row[self.sample_column]),
            "image_path": str(path),
        }
