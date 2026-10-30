"""Select binary operating thresholds from CystoDS calib predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdg_cysto.eval.thresholding import (  # noqa: E402
    binary_confusion_at_threshold,
    select_standard_thresholds,
)
from cdg_cysto.utils.io import write_json, write_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-high-npv", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.calib_predictions)
    if "prob_1" not in frame.columns:
        msg = "calib predictions must contain prob_1 for binary threshold selection."
        raise ValueError(msg)
    labels = frame["y_true"].to_numpy(dtype=int)
    scores = frame["prob_1"].to_numpy(dtype=float)
    thresholds = select_standard_thresholds(labels, scores)
    if not args.include_high_npv:
        thresholds = {
            key: thresholds[key]
            for key in ("default_0.5", "youden", "sensitivity_90")
            if key in thresholds
        }
    rows = []
    for name, threshold in thresholds.items():
        row = {"threshold_name": name}
        row.update(binary_confusion_at_threshold(labels, scores, threshold))
        rows.append(row)
    write_json(thresholds, output / "thresholds.json")
    write_table(pd.DataFrame(rows), output / "threshold_metrics.csv")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
