from __future__ import annotations

import pandas as pd
import pytest

from cdg_cysto.data.metadata_schema import (
    CYSTODS_CANCER_BINARY,
    CYSTODS_FIVE_CLASS,
    EBTC_CANCER_BINARY,
    EBTC_FOUR_CLASS,
    MASTER_METADATA_COLUMNS,
    align_master_metadata,
    validate_master_metadata,
)


def test_fixed_label_mappings() -> None:
    assert EBTC_FOUR_CLASS == {"HGC": 0, "LGC": 1, "NTL": 2, "NST": 3}
    assert EBTC_CANCER_BINARY["HGC"] == 1
    assert EBTC_CANCER_BINARY["NST"] == 0
    assert CYSTODS_FIVE_CLASS["malignant"] == 0
    assert CYSTODS_CANCER_BINARY["normal mucosa"] == 0


def test_align_master_metadata_adds_columns() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["EBTC_000001"],
            "dataset": ["EBTC"],
            "image_path": ["data/raw/EBTC/image.png"],
            "split": ["train"],
        }
    )
    aligned = align_master_metadata(frame)
    assert list(aligned.columns) == MASTER_METADATA_COLUMNS
    assert pd.isna(aligned.loc[0, "mask_path"])


def test_validate_master_metadata_rejects_unknown_dataset() -> None:
    frame = pd.DataFrame([dict.fromkeys(MASTER_METADATA_COLUMNS, pd.NA)])
    frame.loc[0, "dataset"] = "Other"
    with pytest.raises(ValueError):
        validate_master_metadata(frame)
