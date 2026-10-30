"""Canonical metadata schema shared by EBTC and CystoDS."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MASTER_METADATA_PATH = Path("data/processed/master_metadata.csv")

MASTER_METADATA_COLUMNS = [
    "sample_id",
    "dataset",
    "image_path",
    "mask_path",
    "has_mask",
    "label_original",
    "label_ebtc4",
    "label_cancer_binary",
    "label_roi_binary",
    "modality",
    "split",
    "pid",
    "visit",
    "roi_id",
    "width",
    "height",
    "is_valid",
    "error_reason",
]

EBTC_FOUR_CLASS = {
    "HGC": 0,
    "LGC": 1,
    "NTL": 2,
    "NST": 3,
}

EBTC_CANCER_BINARY = {
    "HGC": 1,
    "LGC": 1,
    "NTL": 0,
    "NST": 0,
}

EBTC_ROI_BINARY = {
    "HGC": 1,
    "LGC": 1,
    "NTL": 1,
    "NST": 0,
}

CYSTODS_FIVE_CLASS = {
    "malignant": 0,
    "non-malignant": 1,
    "anatomical landmarks": 2,
    "foreign bodies": 3,
    "normal mucosa": 4,
}

CYSTODS_CANCER_BINARY = {
    "malignant": 1,
    "non-malignant": 0,
    "anatomical landmarks": 0,
    "foreign bodies": 0,
    "normal mucosa": 0,
}

CYSTODS_ROI_BINARY = {
    "malignant": 1,
    "non-malignant": 1,
    "anatomical landmarks": 0,
    "foreign bodies": 0,
    "normal mucosa": 0,
}

CYSTODS_REQUIRED_RAW_COLUMNS = [
    "filename",
    "pid",
    "visit",
    "ROI",
    "bca",
    "class",
    "subclass",
    "stage",
    "modality",
    "JSON",
]


def empty_master_metadata() -> pd.DataFrame:
    return pd.DataFrame(columns=MASTER_METADATA_COLUMNS)


def validate_master_metadata(frame: pd.DataFrame) -> None:
    missing = [column for column in MASTER_METADATA_COLUMNS if column not in frame.columns]
    if missing:
        msg = f"Master metadata is missing columns: {missing}"
        raise ValueError(msg)

    allowed_datasets = {"EBTC", "CystoDS"}
    invalid_datasets = set(frame["dataset"].dropna().astype(str)) - allowed_datasets
    if invalid_datasets:
        msg = f"Unknown dataset values in master metadata: {sorted(invalid_datasets)}"
        raise ValueError(msg)

    allowed_splits = {"train", "val", "test", "external", "unknown"}
    invalid_splits = set(frame["split"].fillna("unknown").astype(str).str.lower()) - allowed_splits
    if invalid_splits:
        msg = f"Unknown split values in master metadata: {sorted(invalid_splits)}"
        raise ValueError(msg)


def align_master_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    aligned = frame.copy()
    for column in MASTER_METADATA_COLUMNS:
        if column not in aligned.columns:
            aligned[column] = pd.NA
    aligned = aligned[MASTER_METADATA_COLUMNS]
    validate_master_metadata(aligned)
    return aligned
