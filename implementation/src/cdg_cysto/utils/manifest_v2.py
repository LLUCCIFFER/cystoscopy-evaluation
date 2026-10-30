"""V2 provenance manifests for reproducible CDG-Cysto runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cdg_cysto.config.schema import save_resolved_config
from cdg_cysto.utils.io import ensure_dir, write_json

V2_REQUIRED_RUN_KEYS = ("run_id", "config_hash", "split_hash", "git_commit", "seed", "command")


def stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_config(config: dict[str, Any]) -> str:
    return hash_text(stable_json(config))


def hash_paths(paths: list[str | Path] | tuple[str | Path, ...] | None) -> str:
    if not paths:
        return hash_text("no-splits")
    payload: list[dict[str, str]] = []
    for path_like in paths:
        path = Path(path_like)
        payload.append(
            {
                "path": str(path),
                "sha256": hash_file(path) if path.exists() and path.is_file() else "missing",
            }
        )
    return hash_text(stable_json(payload))


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def build_run_id(config: dict[str, Any], seed: int) -> str:
    name = str(config.get("experiment_name") or config.get("name") or "v2_run")
    suffix = hash_config({"name": name, "seed": seed, "config": config})[:10]
    safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
    return f"{safe_name}_seed{seed}_{suffix}"


def build_run_manifest(
    config: dict[str, Any],
    *,
    seed: int,
    command: str,
    split_paths: list[str | Path] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "v2.0",
        "run_id": run_id or build_run_id(config, seed),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": hash_config(config),
        "split_hash": hash_paths(split_paths),
        "git_commit": git_commit(),
        "seed": int(seed),
        "command": command,
    }
    return manifest


def validate_run_manifest(manifest: dict[str, Any]) -> None:
    missing = [key for key in V2_REQUIRED_RUN_KEYS if key not in manifest]
    if missing:
        msg = f"run_manifest is missing required keys: {missing}"
        raise ValueError(msg)


def write_run_manifest(
    run_dir: str | Path,
    config: dict[str, Any],
    *,
    seed: int,
    command: str,
    split_paths: list[str | Path] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    run_path = ensure_dir(run_dir)
    save_resolved_config(config, run_path / "config_resolved.yaml")
    manifest = build_run_manifest(
        config,
        seed=seed,
        command=command,
        split_paths=split_paths,
        run_id=run_id,
    )
    validate_run_manifest(manifest)
    write_json(manifest, run_path / "run_manifest.json")
    return manifest


def build_dataset_manifest(metadata_paths: dict[str, str | Path]) -> dict[str, Any]:
    datasets = []
    for name, path_like in sorted(metadata_paths.items()):
        path = Path(path_like)
        datasets.append(
            {
                "dataset": name,
                "metadata_path": str(path),
                "metadata_sha256": hash_file(path) if path.exists() else "missing",
            }
        )
    return {
        "schema_version": "v2.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "manifest_hash": hash_text(stable_json(datasets)),
    }


def write_dataset_manifest(metadata_paths: dict[str, str | Path], output_path: str | Path) -> dict[str, Any]:
    manifest = build_dataset_manifest(metadata_paths)
    write_json(manifest, output_path)
    return manifest


def build_split_manifest(split_paths: dict[str, str | Path]) -> dict[str, Any]:
    splits = []
    for name, path_like in sorted(split_paths.items()):
        path = Path(path_like)
        splits.append(
            {
                "split": name,
                "path": str(path),
                "sha256": hash_file(path) if path.exists() else "missing",
            }
        )
    return {
        "schema_version": "v2.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_hash": hash_text(stable_json(splits)),
        "splits": splits,
    }


def write_split_manifest(split_paths: dict[str, str | Path], output_path: str | Path) -> dict[str, Any]:
    manifest = build_split_manifest(split_paths)
    write_json(manifest, output_path)
    return manifest
