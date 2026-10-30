from __future__ import annotations

import pandas as pd

from cdg_cysto.data.split_v2 import (
    assert_each_split_has_both_classes,
    assert_no_patient_leakage_v2,
    assign_patient_splits,
)


def test_v2_patient_split_has_no_leakage_and_both_classes() -> None:
    rows = []
    for index in range(40):
        rows.append(
            {
                "sample_id": f"s{index}",
                "pid": f"p{index}",
                "label_cancer_binary": index % 2,
                "is_valid": True,
            }
        )
    frame = pd.DataFrame(rows)
    split = assign_patient_splits(frame, seed=3)
    assert_no_patient_leakage_v2(split)
    assert_each_split_has_both_classes(split)
    assert set(split["split_v2"]) == {"train", "val", "calib", "test_internal"}
