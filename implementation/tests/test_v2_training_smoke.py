from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from cdg_cysto.train.engine_v2 import run_v2_experiment  # noqa: E402


def _write_image(path: Path, value: int) -> None:
    image = Image.new("RGB", (32, 32), (value, value, value))
    image.save(path)


def test_v2_training_smoke(tmp_path: Path) -> None:
    rows = []
    for split in ("train", "val"):
        for index in range(4):
            label = index % 2
            image_path = tmp_path / f"{split}_{index}.png"
            _write_image(image_path, 50 + index * 20)
            rows.append(
                {
                    "sample_id": f"{split}_{index}",
                    "dataset": "unit",
                    "image_path": str(image_path),
                    "mask_path": "",
                    "has_mask": False,
                    "pid": f"{split}_p{index}",
                    "visit": "",
                    "roi_id": "",
                    "modality": "WLC",
                    "label_original": str(label),
                    "label_cancer_binary": label,
                    "label_roi_binary": label,
                    "label_five_class": label,
                    "split_source": split,
                    "width": 32,
                    "height": 32,
                    "is_valid": True,
                    "error_reason": "",
                    "image_md5": "",
                }
            )
    frame = pd.DataFrame(rows)
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    frame[frame["split_source"] == "train"].to_csv(train_csv, index=False)
    frame[frame["split_source"] == "val"].to_csv(val_csv, index=False)
    run_dir = tmp_path / "run"
    metrics = run_v2_experiment(
        {
            "experiment_name": "smoke",
            "seed": 1,
            "task": {
                "label_column": "label_cancer_binary",
                "num_classes": 2,
                "classes": ["non_cancer", "cancer"],
            },
            "data": {"train_csv": str(train_csv), "val_csv": str(val_csv)},
            "model": {"name": "tiny_cnn", "num_classes": 2, "input_size": 32},
            "augmentation": {"policy": "basic"},
            "training": {
                "epochs": 1,
                "patience": 1,
                "batch_size": 2,
                "learning_rate": 0.001,
                "prefer_cuda": False,
            },
            "imbalance": {
                "sampler": {"enabled": True, "mode": "inverse_frequency"},
                "loss": {"name": "balanced_softmax"},
            },
            "outputs": {"run_dir": str(run_dir)},
        },
        command="pytest",
    )
    assert "selection_score" in metrics
    for filename in (
        "metrics_epoch.csv",
        "metrics_val.json",
        "predictions_val.csv",
        "logits_val.csv",
        "checkpoint_best.pt",
        "run_manifest.json",
        "class_counts.json",
        "class_priors.json",
        "sampled_class_distribution_epoch.csv",
    ):
        assert (run_dir / filename).exists()
