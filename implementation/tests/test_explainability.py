from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cdg_cysto.explain.cam_metrics import (
    cam_iou_top_percent,
    load_cystods_json_mask,
    roi_energy,
    summarize_cam_against_mask,
)


def test_cystods_json_mask_and_cam_metrics(tmp_path: Path) -> None:
    json_path = tmp_path / "mask.json"
    json_path.write_text(
        json.dumps({"shapes": [{"points": [[1, 1], [4, 1], [4, 4], [1, 4]]}]}),
        encoding="utf-8",
    )
    mask = load_cystods_json_mask(json_path, width=8, height=8)
    cam = np.zeros((8, 8), dtype=float)
    cam[1:5, 1:5] = 1.0
    assert mask.sum() > 0
    assert roi_energy(cam, mask) > 0.8
    assert cam_iou_top_percent(cam, mask) > 0.0
    summary = summarize_cam_against_mask(cam, mask)
    assert set(summary) == {"roi_energy", "pointing_game", "cam_iou_top20"}
