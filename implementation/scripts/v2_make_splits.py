"""Create CDG-Cysto v2 patient-level splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdg_cysto.data.split_v2 import make_v2_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cystods-metadata", default="outputs/v2/manifests/cystods_metadata.csv")
    parser.add_argument("--ebtc-metadata", default="outputs/v2/manifests/ebtc_metadata.csv")
    parser.add_argument("--output-dir", default="outputs/v2/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", default="0.60,0.15,0.10,0.15")
    args = parser.parse_args()

    ratios = tuple(float(value.strip()) for value in args.ratios.split(","))
    if len(ratios) != 4:
        msg = "--ratios must contain four comma-separated values"
        raise ValueError(msg)
    result = make_v2_splits(
        args.cystods_metadata,
        args.ebtc_metadata,
        output_dir=args.output_dir,
        seed=args.seed,
        ratios=ratios,  # type: ignore[arg-type]
    )
    print(f"Wrote split files to {args.output_dir}")
    print(f"Wrote split manifest: {result['split_manifest']}")
    print(f"Wrote split audit: {result['audit_path']}")


if __name__ == "__main__":
    main()
