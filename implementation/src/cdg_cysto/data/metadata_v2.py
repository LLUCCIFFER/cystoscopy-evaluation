"""CDG-Cysto v2 metadata normalization."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from cdg_cysto.data.cystods import build_cystods_metadata
from cdg_cysto.data.ebtc import build_ebtc_metadata
from cdg_cysto.data.metadata_schema import CYSTODS_FIVE_CLASS
from cdg_cysto.utils.io import write_table

V2_METADATA_COLUMNS = [
    "sample_id",
    "dataset",
    "image_path",
    "mask_path",
    "has_mask",
    "pid",
    "visit",
    "roi_id",
    "modality",
    "label_original",
    "label_cancer_binary",
    "label_roi_binary",
    "label_five_class",
    "split_source",
    "width",
    "height",
    "is_valid",
    "error_reason",
    "image_md5",
]


def file_md5(path: str | Path) -> str:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return ""
    digest = hashlib.md5()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cystods_label_five_class(label: object) -> int | None:
    text = str(label).strip().lower()
    if text not in CYSTODS_FIVE_CLASS:
        return None
    return int(CYSTODS_FIVE_CLASS[text])


def align_v2_metadata(frame: pd.DataFrame, *, compute_md5: bool = True) -> pd.DataFrame:
    aligned = frame.copy()
    if "split_source" not in aligned.columns:
        aligned["split_source"] = aligned.get("split", "unknown")
    if "label_five_class" not in aligned.columns:
        aligned["label_five_class"] = [
            cystods_label_five_class(label) if dataset == "CystoDS" else pd.NA
            for label, dataset in zip(aligned["label_original"], aligned["dataset"], strict=False)
        ]
    if "image_md5" not in aligned.columns:
        if compute_md5:
            aligned["image_md5"] = [file_md5(path) for path in aligned["image_path"]]
        else:
            aligned["image_md5"] = ""
    for column in V2_METADATA_COLUMNS:
        if column not in aligned.columns:
            aligned[column] = pd.NA
    return aligned[V2_METADATA_COLUMNS].reset_index(drop=True)


def build_cystods_v2_metadata(
    raw_root: str | Path,
    source_metadata: str | Path | None = None,
    *,
    compute_md5: bool = True,
) -> pd.DataFrame:
    frame = build_cystods_metadata(raw_root, source_metadata, split_by_patient=False)
    return align_v2_metadata(frame, compute_md5=compute_md5)


def build_ebtc_v2_metadata(
    raw_root: str | Path,
    source_metadata: str | Path | None = None,
    *,
    compute_md5: bool = True,
) -> pd.DataFrame:
    frame = build_ebtc_metadata(raw_root, source_metadata)
    return align_v2_metadata(frame, compute_md5=compute_md5)


def write_data_audit(
    frames: dict[str, pd.DataFrame],
    output_path: str | Path,
) -> None:
    def counts_markdown(series: pd.Series) -> str:
        counts = series.value_counts(dropna=False)
        rows = ["| value | count |", "| --- | --- |"]
        for value, count in counts.items():
            rows.append(f"| {value} | {int(count)} |")
        return "\n".join(rows)

    lines: list[str] = ["# Stage 01 Data Audit", ""]
    for dataset, frame in frames.items():
        lines.extend([f"## {dataset}", ""])
        lines.append(f"- rows: {len(frame)}")
        lines.append(f"- valid_rows: {int(frame['is_valid'].astype(bool).sum())}")
        lines.append(f"- patients: {frame['pid'].astype(str).nunique()}")
        lines.append(f"- masks: {int(frame['has_mask'].astype(bool).sum())}")
        lines.extend(["", "### Class Distribution", ""])
        lines.append(counts_markdown(frame["label_original"]))
        lines.extend(["", "### Modality Distribution", ""])
        lines.append(counts_markdown(frame["modality"]))
        broken = frame[~frame["is_valid"].astype(bool)]
        lines.extend(["", "### Broken Images", ""])
        if broken.empty:
            lines.append("_None detected._")
        else:
            rows = ["| sample_id | image_path | error_reason |", "| --- | --- | --- |"]
            for _, row in broken[["sample_id", "image_path", "error_reason"]].head(200).iterrows():
                rows.append(f"| {row['sample_id']} | {row['image_path']} | {row['error_reason']} |")
            lines.append("\n".join(rows))
        lines.append("")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def prepare_v2_metadata(
    *,
    cystods_raw_root: str | Path,
    ebtc_raw_root: str | Path,
    output_dir: str | Path = "outputs/v2/manifests",
    cystods_source_metadata: str | Path | None = None,
    ebtc_source_metadata: str | Path | None = None,
    compute_md5: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cystods = build_cystods_v2_metadata(
        cystods_raw_root,
        cystods_source_metadata,
        compute_md5=compute_md5,
    )
    ebtc = build_ebtc_v2_metadata(ebtc_raw_root, ebtc_source_metadata, compute_md5=compute_md5)
    cystods_path = output / "cystods_metadata.csv"
    ebtc_path = output / "ebtc_metadata.csv"
    write_table(cystods, cystods_path)
    write_table(ebtc, ebtc_path)
    audit_path = output / "stage01_data_audit.md"
    write_data_audit({"CystoDS": cystods, "EBTC": ebtc}, audit_path)
    return {
        "cystods": cystods,
        "ebtc": ebtc,
        "cystods_path": cystods_path,
        "ebtc_path": ebtc_path,
        "audit_path": audit_path,
    }
