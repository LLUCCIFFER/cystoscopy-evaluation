"""Dataset label mappings used by CDG-Cysto experiments."""

from __future__ import annotations

from collections.abc import Iterable

EBTC_CLASSES = ("HGC", "LGC", "NTL", "NST")
BINARY_CANCER_CLASSES = ("non_cancer", "cancer")

EBTC_TO_CANCER = {
    "HGC": "cancer",
    "LGC": "cancer",
    "NTL": "non_cancer",
    "NST": "non_cancer",
}

EBTC_TO_ROI = {
    "HGC": "roi",
    "LGC": "roi",
    "NTL": "roi",
    "NST": "non_roi",
}

CYSTODS_CLASSES = (
    "malignant",
    "non-malignant",
    "anatomical landmarks",
    "foreign bodies",
    "normal mucosa",
)

CANCER_ALIASES = {
    "HGC",
    "LGC",
    "HIGH_GRADE_CARCINOMA",
    "LOW_GRADE_CARCINOMA",
    "CANCER",
    "CANCEROUS",
    "TUMOR",
    "TUMOUR",
    "NEOPLASM",
    "MALIGNANT",
    "MALIGNANCY",
    "BLADDER_CANCER",
}

NON_CANCER_ALIASES = {
    "NTL",
    "NST",
    "NORMAL",
    "NON_CANCER",
    "NONCANCER",
    "BENIGN",
    "HEALTHY",
    "INFLAMMATION",
    "STONE",
    "STENT",
    "ARTIFACT",
    "ARTEFACT",
}


def normalize_label(label: object) -> str:
    """Normalize labels to a stable uppercase token."""

    text = str(label).strip()
    text = text.replace("-", "_").replace("/", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.upper()


def canonical_ebtc_label(label: object) -> str:
    normalized = normalize_label(label)
    if normalized not in EBTC_CLASSES:
        msg = f"Unknown EBTC label '{label}'. Expected one of {EBTC_CLASSES}."
        raise ValueError(msg)
    return normalized


def map_ebtc_to_cancer_binary(label: object) -> str:
    return EBTC_TO_CANCER[canonical_ebtc_label(label)]


def map_cancer_binary(label: object) -> str:
    normalized = normalize_label(label)
    if normalized in CANCER_ALIASES:
        return "cancer"
    if normalized in NON_CANCER_ALIASES:
        return "non_cancer"
    msg = f"Cannot map label '{label}' to cancer/non_cancer. Add it to label_mapping.py."
    raise ValueError(msg)


def encode_labels(labels: Iterable[object], classes: Iterable[str]) -> list[int]:
    class_list = list(classes)
    index = {normalize_label(name): idx for idx, name in enumerate(class_list)}
    encoded: list[int] = []
    for label in labels:
        normalized = normalize_label(label)
        if normalized not in index:
            msg = f"Unknown label '{label}'. Expected one of {class_list}."
            raise ValueError(msg)
        encoded.append(index[normalized])
    return encoded


def decode_labels(indices: Iterable[int], classes: Iterable[str]) -> list[str]:
    class_list = list(classes)
    decoded: list[str] = []
    for index in indices:
        if index < 0 or index >= len(class_list):
            msg = f"Class index {index} is outside 0..{len(class_list) - 1}."
            raise ValueError(msg)
        decoded.append(class_list[index])
    return decoded


def validate_label_set(labels: Iterable[object], classes: Iterable[str]) -> None:
    encode_labels(labels, classes)
