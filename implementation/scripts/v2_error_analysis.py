"""Generate CDG-Cysto v2 error analysis artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdg_cysto.analysis.error_analysis import (  # noqa: E402
    copy_case_images,
    error_by_group,
    merge_predictions_metadata,
    select_error_cases,
    summarize_errors,
)
from cdg_cysto.utils.io import write_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    metadata = pd.read_csv(args.metadata)
    merged = merge_predictions_metadata(predictions, metadata)
    write_table(summarize_errors(merged), output / "error_summary.csv")
    write_table(error_by_group(merged, "label_original"), output / "error_by_class.csv")
    write_table(error_by_group(merged, "modality"), output / "error_by_modality.csv")
    if set(merged["label_original"].dropna().astype(str)).issubset({"HGC", "LGC", "NTL", "NST"}):
        write_table(error_by_group(merged, "label_original"), output / "error_by_ebtc_class.csv")
    cases = select_error_cases(merged, top_k=args.top_k)
    for name, frame in cases.items():
        write_table(frame, output / f"{name}.csv")
    copy_case_images(
        cases["top_false_negatives"],
        output / "top_false_negatives",
        prefix="fn",
    )
    copy_case_images(
        cases["top_false_positives"],
        output / "top_false_positives",
        prefix="fp",
    )
    print(f"Wrote error analysis to {output}")


if __name__ == "__main__":
    main()
