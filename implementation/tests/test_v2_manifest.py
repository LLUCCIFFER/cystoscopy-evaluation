from __future__ import annotations

from cdg_cysto.utils.manifest_v2 import V2_REQUIRED_RUN_KEYS, build_run_manifest


def test_v2_run_manifest_required_keys() -> None:
    manifest = build_run_manifest(
        {"experiment_name": "unit_test"},
        seed=7,
        command="pytest",
        split_paths=[],
    )
    assert set(V2_REQUIRED_RUN_KEYS).issubset(manifest)
    assert manifest["seed"] == 7
    assert manifest["command"] == "pytest"
