"""Prepare standardized CDG-Cysto v2 metadata for CystoDS and EBTC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdg_cysto.data.metadata_v2 import prepare_v2_metadata  # noqa: E402
from cdg_cysto.utils.manifest_v2 import write_dataset_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cystods-raw-root", default="data/raw/CystoDS")
    parser.add_argument("--ebtc-raw-root", default="data/raw/EBTC")
    parser.add_argument("--cystods-source-metadata")
    parser.add_argument("--ebtc-source-metadata")
    parser.add_argument("--output-dir", default="outputs/v2/manifests")
    parser.add_argument("--skip-md5", action="store_true")
    args = parser.parse_args()

    result = prepare_v2_metadata(
        cystods_raw_root=args.cystods_raw_root,
        ebtc_raw_root=args.ebtc_raw_root,
        cystods_source_metadata=args.cystods_source_metadata,
        ebtc_source_metadata=args.ebtc_source_metadata,
        output_dir=args.output_dir,
        compute_md5=not args.skip_md5,
    )
    manifest_path = Path(args.output_dir) / "dataset_manifest.json"
    write_dataset_manifest(
        {
            "CystoDS": result["cystods_path"],
            "EBTC": result["ebtc_path"],
        },
        manifest_path,
    )
    print(f"Wrote CystoDS metadata: {result['cystods_path']} ({len(result['cystods'])} rows)")
    print(f"Wrote EBTC metadata: {result['ebtc_path']} ({len(result['ebtc'])} rows)")
    print(f"Wrote data audit: {result['audit_path']}")
    print(f"Wrote dataset manifest: {manifest_path}")


if __name__ == "__main__":
    main()
