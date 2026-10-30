from __future__ import annotations

import pandas as pd

from cdg_cysto.data.metadata_v2 import V2_METADATA_COLUMNS, align_v2_metadata


def test_v2_metadata_schema_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "dataset": "CystoDS",
                "image_path": "missing.png",
                "mask_path": "",
                "has_mask": False,
                "pid": "p1",
                "visit": "",
                "roi_id": "",
                "modality": "WLC",
                "label_original": "malignant",
                "label_cancer_binary": 1,
                "label_roi_binary": 1,
                "split": "external",
                "width": 10,
                "height": 10,
                "is_valid": False,
                "error_reason": "missing",
            }
        ]
    )
    aligned = align_v2_metadata(frame, compute_md5=False)
    assert list(aligned.columns) == V2_METADATA_COLUMNS
    assert aligned.loc[0, "label_five_class"] == 0
    assert aligned.loc[0, "split_source"] == "external"
