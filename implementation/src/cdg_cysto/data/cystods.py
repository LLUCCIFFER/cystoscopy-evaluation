"""CystoDS metadata preparation and loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from cdg_cysto.data.metadata_schema import (
    CYSTODS_CANCER_BINARY,
    CYSTODS_ROI_BINARY,
    MASTER_METADATA_COLUMNS,
    align_master_metadata,
)
from cdg_cysto.utils.io import read_table, write_table

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
REQUIRED_COLUMNS = set(MASTER_METADATA_COLUMNS)
CYSTODS_CLASS_CANONICAL = {
    "malignant": "malignant",
    "non-malignant": "non-malignant",
    "anatomical landmarks": "anatomical landmarks",
    "foreign bodies": "foreign bodies",
    "normal mucosa": "normal mucosa",
}


def find_cystods_root(raw_root: str | Path) -> Path:
    """Return the directory containing cystods.csv and image folders."""

    root = Path(raw_root)
    if (root / "cystods.csv").exists():
        return root
    candidate = root / "osf_project"
    if (candidate / "cystods.csv").exists():
        return candidate
    matches = sorted(root.glob("**/cystods.csv"))
    if matches:
        return matches[0].parent
    msg = f"Could not find cystods.csv under {root}."
    raise FileNotFoundError(msg)


def _canonical_class(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized not in CYSTODS_CLASS_CANONICAL:
        msg = f"Unknown CystoDS class '{value}'."
        raise ValueError(msg)
    return CYSTODS_CLASS_CANONICAL[normalized]


def _image_dir(root: Path) -> Path:
    for name in ("images", "image"):
        candidate = root / name
        if candidate.exists():
            return candidate
    return root / "images"


def _segmentation_dir(root: Path) -> Path:
    for name in ("segmentations", "segmentation"):
        candidate = root / name
        if candidate.exists():
            return candidate
    return root / "segmentations"


def _resolve_image_path(root: Path, filename: str) -> Path:
    stem = Path(filename).stem
    image_root = _image_dir(root)
    for suffix in (Path(filename).suffix, ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        if not suffix:
            continue
        candidate = image_root / f"{stem}{suffix.lower()}"
        if candidate.exists():
            return candidate
    return image_root / f"{stem}.png"


def _resolve_mask_path(root: Path, filename: str) -> Path:
    return _segmentation_dir(root) / f"{Path(filename).stem}.json"


def load_cystods_metadata(
    metadata_csv: str | Path,
    *,
    check_leakage: bool = True,
    binary: bool = True,
) -> pd.DataFrame:
    frame = read_table(metadata_csv)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        msg = f"CystoDS metadata is missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    frame = frame.copy()
    frame["pid"] = frame["pid"].astype(str)
    if "split" not in frame.columns:
        frame["split"] = "external"
    else:
        frame["split"] = frame["split"].astype(str).str.lower()

    if binary:
        frame["target"] = frame["label_cancer_binary"].astype(int)

    if check_leakage and frame["split"].nunique() > 1:
        from cdg_cysto.data.split import assert_no_patient_leakage

        assert_no_patient_leakage(frame, patient_column="pid")
    return frame


def scan_cystods_folders(raw_root: str | Path) -> pd.DataFrame:
    root = find_cystods_root(raw_root)
    rows: list[dict[str, str]] = []
    for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for image_path in sorted(label_dir.glob("**/*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rows.append(
                {
                    "image_path": str(image_path.relative_to(root)),
                    "label": label_dir.name,
                    "patient_id": "unknown",
                    "split": "test",
                }
            )
    return pd.DataFrame(rows)


def build_cystods_metadata(
    raw_root: str | Path,
    source_metadata: str | Path | None = None,
    *,
    split_by_patient: bool = True,
    seed: int = 42,
    train_size: float = 0.70,
    val_size: float = 0.15,
) -> pd.DataFrame:
    """Build canonical CystoDS metadata from cystods.csv."""

    root = find_cystods_root(raw_root)
    source = Path(source_metadata) if source_metadata is not None else root / "cystods.csv"
    raw = read_table(source)
    required = {"filename", "pid", "visit", "bca", "class", "modality", "json"}
    missing = required - set(raw.columns)
    if missing:
        msg = f"CystoDS metadata is missing required raw columns: {sorted(missing)}"
        raise ValueError(msg)

    split_map: dict[str, str] = {}
    if split_by_patient:
        from cdg_cysto.data.split import make_patient_level_split

        split_frame = make_patient_level_split(
            raw.rename(columns={"class": "label_original"}),
            patient_column="pid",
            label_column="label_original",
            seed=seed,
            train_size=train_size,
            val_size=val_size,
        )
        split_map = dict(zip(split_frame["pid"].astype(str), split_frame["split"], strict=False))

    rows: list[dict[str, object]] = []
    for _, row in raw.iterrows():
        filename = str(row["filename"]).strip()
        class_name = _canonical_class(row["class"])
        image_path = _resolve_image_path(root, filename)
        mask_path = _resolve_mask_path(root, filename)
        has_mask = bool(int(row["json"]) == 1 and mask_path.exists())
        pid = str(row["pid"]).strip()

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
                "sample_id": f"CystoDS_{Path(filename).stem}",
                "dataset": "CystoDS",
                "image_path": str(image_path),
                "mask_path": str(mask_path) if has_mask else "",
                "has_mask": has_mask,
                "label_original": class_name,
                "label_ebtc4": pd.NA,
                "label_cancer_binary": CYSTODS_CANCER_BINARY[class_name],
                "label_roi_binary": CYSTODS_ROI_BINARY[class_name],
                "modality": str(row["modality"]).strip(),
                "split": split_map.get(pid, "external"),
                "pid": pid,
                "visit": "" if pd.isna(row["visit"]) else str(row["visit"]),
                "roi_id": "" if "lesion" not in raw.columns or pd.isna(row["lesion"]) else str(row["lesion"]),
                "width": width,
                "height": height,
                "is_valid": is_valid,
                "error_reason": error_reason,
            }
        )

    return align_master_metadata(pd.DataFrame(rows))


def prepare_cystods_metadata(
    raw_root: str | Path,
    output_csv: str | Path,
    source_metadata: str | Path | None = None,
    *,
    split_by_patient: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    frame = build_cystods_metadata(
        raw_root,
        source_metadata,
        split_by_patient=split_by_patient,
        seed=seed,
    )
    write_table(frame, output_csv)
    return frame
