from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

from cdg_cysto.data.cystods import build_cystods_metadata
from cdg_cysto.data.ebtc import build_ebtc_metadata
from cdg_cysto.data.metadata_schema import MASTER_METADATA_COLUMNS


def test_build_ebtc_metadata_from_official_csv(tmp_path: Path) -> None:
    root = tmp_path / "baldder_tissue_classification"
    image_dir = root / "HGC"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (16, 12)).save(image_dir / "case_001_pt_002_frame_0003.png")
    pd.DataFrame(
        {
            "HLY": ["case_001_pt_002_frame_0003.png"],
            "imaging type": ["WLI"],
            "tissue type": ["HGC"],
            "sub_dataset": ["train"],
        }
    ).to_csv(root / "annotations.csv", index=False)

    frame = build_ebtc_metadata(tmp_path)
    assert frame.columns.tolist() == MASTER_METADATA_COLUMNS
    assert frame.loc[0, "label_ebtc4"] == 0
    assert frame.loc[0, "label_cancer_binary"] == 1
    assert frame.loc[0, "pid"] == "pt_002"
    assert frame.loc[0, "is_valid"]


def test_build_cystods_metadata_from_csv_and_json(tmp_path: Path) -> None:
    root = tmp_path / "osf_project"
    (root / "images").mkdir(parents=True)
    (root / "segmentations").mkdir()
    Image.new("RGB", (20, 10)).save(root / "images" / "abc.png")
    (root / "segmentations" / "abc.json").write_text(
        json.dumps({"shapes": [{"points": [[1, 1], [5, 1], [5, 5], [1, 5]]}]}),
        encoding="utf-8",
    )
    rows = []
    for patient in range(12):
        stem = f"abc{patient}"
        Image.new("RGB", (20, 10)).save(root / "images" / f"{stem}.png")
        rows.append(
            {
                "filename": f"{stem}.png",
                "pid": patient,
                "visit": "",
                "lesion": "",
                "multifocal": "",
                "bca": "Yes" if patient % 2 else "No",
                "class": "Malignant" if patient % 2 else "Normal mucosa",
                "subclass": "",
                "subclass2": "",
                "stage": "",
                "morphology": "",
                "modality": "WLC",
                "json": 0,
            }
        )
    rows[0]["filename"] = "abc.png"
    rows[0]["json"] = 1
    pd.DataFrame(rows).to_csv(root / "cystods.csv", index=False)

    frame = build_cystods_metadata(tmp_path)
    assert frame.columns.tolist() == MASTER_METADATA_COLUMNS
    assert frame["pid"].nunique() == 12
    assert set(frame["split"]) == {"train", "val", "test"}
    assert frame.loc[frame["sample_id"] == "CystoDS_abc", "has_mask"].item()
    assert frame["label_cancer_binary"].isin([0, 1]).all()
