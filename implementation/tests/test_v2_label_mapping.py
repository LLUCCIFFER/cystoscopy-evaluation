from __future__ import annotations

from cdg_cysto.data.metadata_schema import (
    CYSTODS_CANCER_BINARY,
    CYSTODS_FIVE_CLASS,
    CYSTODS_ROI_BINARY,
    EBTC_CANCER_BINARY,
    EBTC_ROI_BINARY,
)


def test_v2_cystods_binary_mappings() -> None:
    assert CYSTODS_CANCER_BINARY["malignant"] == 1
    assert CYSTODS_CANCER_BINARY["normal mucosa"] == 0
    assert CYSTODS_ROI_BINARY["malignant"] == 1
    assert CYSTODS_ROI_BINARY["non-malignant"] == 1
    assert CYSTODS_ROI_BINARY["foreign bodies"] == 0
    assert CYSTODS_FIVE_CLASS["malignant"] == 0


def test_v2_ebtc_binary_mappings() -> None:
    assert EBTC_CANCER_BINARY["HGC"] == 1
    assert EBTC_CANCER_BINARY["LGC"] == 1
    assert EBTC_CANCER_BINARY["NTL"] == 0
    assert EBTC_ROI_BINARY["NTL"] == 1
    assert EBTC_ROI_BINARY["NST"] == 0
