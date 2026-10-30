"""Collect raw logits from a frozen CDG-Cysto v2 checkpoint on one split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cdg_cysto.eval.metrics import ensure_probabilities  # noqa: E402
from cdg_cysto.utils.io import write_table  # noqa: E402
from scripts.v2_external_validate import _collect, _predictions, _write_logits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default="label_cancer_binary")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.split_csv)
    metadata = metadata[metadata["is_valid"].astype(bool)].reset_index(drop=True)
    labels = metadata[args.label_column].astype(int).to_numpy()
    logits, sample_ids = _collect(Path(args.checkpoint), metadata, args.label_column)
    probabilities = ensure_probabilities(logits)
    _write_logits(sample_ids, labels, logits, output / "logits.csv")
    write_table(_predictions(sample_ids, labels, probabilities), output / "predictions.csv")
    print(f"Wrote raw logits and predictions to {output}")


if __name__ == "__main__":
    main()
