"""EBTC metadata preparation and loading."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from PIL import Image

from cdg_cysto.data.metadata_schema import (
    EBTC_CANCER_BINARY,
    EBTC_FOUR_CLASS,
    EBTC_ROI_BINARY,
    MASTER_METADATA_COLUMNS,
    align_master_metadata,
)
from cdg_cysto.utils.io import read_table, write_table

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
OFFICIAL_ARCHIVE_MD5 = "f507e28908fa73cb01fcfd48e1951675"
REQUIRED_COLUMNS = set(MASTER_METADATA_COLUMNS)


def find_ebtc_root(raw_root: str | Path) -> Path:
    """Return the directory containing EBTC annotations and class folders."""

    root = Path(raw_root)
    if (root / "annotations.csv").exists():
        return root
    candidate = root / "baldder_tissue_classification"
    if (candidate / "annotations.csv").exists():
        return candidate
    matches = sorted(root.glob("**/annotations.csv"))
    if matches:
        return matches[0].parent
    msg = f"Could not find EBTC annotations.csv under {root}."
    raise FileNotFoundError(msg)


def _parse_pid(filename: str) -> str:
    match = re.search(r"pt_(\d+)", filename)
    return f"pt_{match.group(1)}" if match else "unknown"


def _parse_visit(filename: str) -> str:
    match = re.search(r"case_(\d+)", filename)
    return f"case_{match.group(1)}" if match else "unknown"


def _parse_roi_id(filename: str) -> str:
    match = re.search(r"frame_(\d+)", filename)
    return f"frame_{match.group(1)}" if match else Path(filename).stem


def _resolve_image_path(root: Path, filename: str, label: str) -> Path:
    direct = root / label / filename
    if direct.exists():
        return direct
    alternatives = {
        filename.replace("_NST_", "_HLT_"),
        filename.replace("_NST__", "_HLT__"),
        filename.replace("NST_frame", "HLT_frame"),
    }
    for alternative in sorted(alternatives):
        candidate = root / label / alternative
        if candidate.exists():
            return candidate
    matches = list(root.glob(f"*/{filename}"))
    if matches:
        return matches[0]
    for alternative in sorted(alternatives):
        matches = list(root.glob(f"*/{alternative}"))
        if matches:
            return matches[0]
    return direct


def load_ebtc_metadata(
    metadata_csv: str | Path,
    *,
    check_leakage: bool = True,
    binary: bool = False,
) -> pd.DataFrame:
    frame = read_table(metadata_csv)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        msg = f"EBTC metadata is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    frame = frame.copy()
    frame["pid"] = frame["pid"].astype(str)
    frame["split"] = frame["split"].astype(str).str.lower()
    if binary:
        frame["target"] = frame["label_cancer_binary"].astype(int)
    else:
        frame["target"] = frame["label_ebtc4"].astype(int)

    if check_leakage:
        from cdg_cysto.data.split import assert_no_patient_leakage

        assert_no_patient_leakage(frame, patient_column="pid")
    return frame


def scan_ebtc_folders(raw_root: str | Path) -> pd.DataFrame:
    """Create provisional metadata by scanning class folders."""

    root = find_ebtc_root(raw_root)
    rows: list[dict[str, object]] = []
    for class_name in EBTC_FOUR_CLASS:
        for image_path in sorted((root / class_name).glob("**/*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rows.append(
                {
                    "HLY": image_path.name,
                    "imaging type": "unknown",
                    "tissue type": class_name,
                    "split": "unknown",
                }
            )
    return pd.DataFrame(rows)


def build_ebtc_metadata(raw_root: str | Path, source_metadata: str | Path | None = None) -> pd.DataFrame:
    """Build canonical EBTC metadata from the official annotation file."""

    root = find_ebtc_root(raw_root)
    source = Path(source_metadata) if source_metadata is not None else root / "annotations.csv"
    annotations = read_table(source)
    if "HLY" not in annotations.columns:
        msg = f"EBTC annotations must contain the HLY filename column: {source}"
        raise ValueError(msg)

    rows: list[dict[str, object]] = []
    for _index, row in annotations.iterrows():
        filename = str(row["HLY"]).strip()
        label = str(row["tissue type"]).strip().upper()
        modality = str(row["imaging type"]).strip().upper()
        split = str(row["sub_dataset"]).strip().lower()
        image_path = _resolve_image_path(root, filename, label)

        is_valid = image_path.exists()
        error_reason = "" if is_valid else "missing_image"
        width: int | None = None
        height: int | None = None
        if is_valid:
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except OSError:
                is_valid = False
                error_reason = "cannot_open_image"

        rows.append(
            {
                "sample_id": f"EBTC_{Path(filename).stem}",
                "dataset": "EBTC",
                "image_path": str(image_path),
                "mask_path": "",
                "has_mask": False,
                "label_original": label,
                "label_ebtc4": EBTC_FOUR_CLASS[label],
                "label_cancer_binary": EBTC_CANCER_BINARY[label],
                "label_roi_binary": EBTC_ROI_BINARY[label],
                "modality": modality,
                "split": split,
                "pid": _parse_pid(filename),
                "visit": _parse_visit(filename),
                "roi_id": _parse_roi_id(filename),
                "width": width,
                "height": height,
                "is_valid": is_valid,
                "error_reason": error_reason,
            }
        )

    frame = align_master_metadata(pd.DataFrame(rows))
    frame.insert(0, "_order", range(len(frame)))
    frame = frame.sort_values(["split", "label_ebtc4", "_order"]).drop(columns="_order")
    return frame.reset_index(drop=True)


def prepare_ebtc_metadata(
    raw_root: str | Path,
    output_csv: str | Path,
    source_metadata: str | Path | None = None,
) -> pd.DataFrame:
    frame = build_ebtc_metadata(raw_root, source_metadata)
    write_table(frame, output_csv)
    return frame
