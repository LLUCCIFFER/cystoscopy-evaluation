from __future__ import annotations

import pytest

from cdg_cysto.data.label_mapping import (
    BINARY_CANCER_CLASSES,
    EBTC_CLASSES,
    decode_labels,
    encode_labels,
    map_cancer_binary,
    map_ebtc_to_cancer_binary,
)


def test_ebtc_binary_mapping() -> None:
    assert map_ebtc_to_cancer_binary("HGC") == "cancer"
    assert map_ebtc_to_cancer_binary("LGC") == "cancer"
    assert map_ebtc_to_cancer_binary("NTL") == "non_cancer"
    assert map_ebtc_to_cancer_binary("NST") == "non_cancer"


def test_binary_class_order_matches_numeric_mapping() -> None:
    assert BINARY_CANCER_CLASSES == ("non_cancer", "cancer")


def test_cancer_alias_mapping() -> None:
    assert map_cancer_binary("tumor") == "cancer"
    assert map_cancer_binary("benign") == "non_cancer"


def test_encode_decode_ebtc_labels() -> None:
    encoded = encode_labels(["HGC", "NST"], EBTC_CLASSES)
    assert encoded == [0, 3]
    assert decode_labels(encoded, EBTC_CLASSES) == ["HGC", "NST"]


def test_unknown_label_raises() -> None:
    with pytest.raises(ValueError):
        encode_labels(["unknown"], EBTC_CLASSES)
