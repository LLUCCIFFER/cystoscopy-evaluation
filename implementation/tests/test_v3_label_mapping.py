from __future__ import annotations

import pytest

from cdg_cysto.data.label_mapping_v3 import (
    V3_CLASS_NAMES,
    map_cystods_label_to_v3,
    map_ebtc_label_to_v3,
    v3_class_name,
)


def test_v3_cystods_three_class_mapping() -> None:
    assert map_cystods_label_to_v3("malignant") == 0
    assert map_cystods_label_to_v3("non-malignant") == 1
    assert map_cystods_label_to_v3("normal mucosa") == 2
    assert map_cystods_label_to_v3("anatomical landmarks") == 2
    assert map_cystods_label_to_v3("foreign bodies") == 2


def test_v3_ebtc_three_class_mapping() -> None:
    assert map_ebtc_label_to_v3("HGC") == 0
    assert map_ebtc_label_to_v3("LGC") == 0
    assert map_ebtc_label_to_v3("NTL") == 1
    assert map_ebtc_label_to_v3("NST") == 2


def test_v3_class_names() -> None:
    assert V3_CLASS_NAMES == ["malignant", "benign_abnormal", "normal"]
    assert v3_class_name(1) == "benign_abnormal"


def test_v3_mapping_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        map_cystods_label_to_v3("unknown")
    with pytest.raises(ValueError):
        map_ebtc_label_to_v3("unknown")
