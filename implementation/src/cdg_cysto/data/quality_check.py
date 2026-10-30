"""Dataset quality checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cdg_cysto.data.split import find_patient_leakage, summarize_splits


def missing_image_paths(
    frame: pd.DataFrame,
    image_root: str | Path,
    image_column: str = "image_path",
) -> list[str]:
    root = Path(image_root)
    missing: list[str] = []
    for value in frame[image_column]:
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            missing.append(str(path))
    return missing


def dataset_quality_report(
    frame: pd.DataFrame,
    image_root: str | Path,
    *,
    split_column: str = "split",
    label_column: str = "label",
    patient_column: str = "patient_id",
    image_column: str = "image_path",
) -> dict[str, object]:
    return {
        "num_images": int(len(frame)),
        "num_patients": int(frame[patient_column].nunique()),
        "split_summary": summarize_splits(frame, split_column, label_column, patient_column).to_dict(
            orient="records"
        ),
        "patient_leakage": {
            f"{left}/{right}": sorted(values)
            for (left, right), values in find_patient_leakage(
                frame, split_column, patient_column
            ).items()
        },
        "missing_images": missing_image_paths(frame, image_root, image_column),
    }
