from __future__ import annotations

import pandas as pd
import pytest

from cdg_cysto.data.split import (
    assert_no_patient_leakage,
    find_patient_leakage,
    make_patient_level_split,
)


def test_no_patient_leakage_passes() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3"],
            "split": ["train", "val", "test"],
            "label": ["HGC", "LGC", "NST"],
        }
    )
    assert_no_patient_leakage(frame)


def test_patient_leakage_detected() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2"],
            "split": ["train", "test", "val"],
            "label": ["HGC", "LGC", "NST"],
        }
    )
    leakage = find_patient_leakage(frame)
    assert ("test", "train") in leakage or ("train", "test") in leakage
    with pytest.raises(ValueError):
        assert_no_patient_leakage(frame)


def test_make_patient_level_split_has_no_pid_leakage() -> None:
    frame = pd.DataFrame(
        {
            "pid": [f"p{patient}" for patient in range(20) for _ in range(2)],
            "label_original": ["malignant" if patient % 2 else "normal mucosa" for patient in range(20) for _ in range(2)],
        }
    )
    split_frame = make_patient_level_split(frame, seed=7)
    assert set(split_frame["split"]) == {"train", "val", "test"}
    assert_no_patient_leakage(split_frame, patient_column="pid")
