"""CDG-Cysto v3 label mappings."""

from __future__ import annotations

V3_THREE_CLASS = {
    "malignant": 0,
    "benign_abnormal": 1,
    "normal": 2,
}

CYSTODS_TO_V3_THREE_CLASS = {
    "malignant": 0,
    "non-malignant": 1,
    "anatomical landmarks": 2,
    "foreign bodies": 2,
    "normal mucosa": 2,
}

EBTC_TO_V3_THREE_CLASS = {
    "HGC": 0,
    "LGC": 0,
    "NTL": 1,
    "NST": 2,
}

V3_CLASS_NAMES = ["malignant", "benign_abnormal", "normal"]


def map_cystods_label_to_v3(label: str) -> int:
    normalized = str(label).strip().lower()
    if normalized not in CYSTODS_TO_V3_THREE_CLASS:
        msg = f"Unsupported CystoDS v3 label: {label}"
        raise ValueError(msg)
    return CYSTODS_TO_V3_THREE_CLASS[normalized]


def map_ebtc_label_to_v3(label: str) -> int:
    normalized = str(label).strip().upper()
    if normalized not in EBTC_TO_V3_THREE_CLASS:
        msg = f"Unsupported EBTC v3 label: {label}"
        raise ValueError(msg)
    return EBTC_TO_V3_THREE_CLASS[normalized]


def v3_class_name(index: int) -> str:
    return V3_CLASS_NAMES[int(index)]
