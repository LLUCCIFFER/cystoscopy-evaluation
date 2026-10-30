"""Patient-level split helpers for CDG-Cysto v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from cdg_cysto.utils.io import write_table
from cdg_cysto.utils.manifest_v2 import write_split_manifest

V2_SPLIT_NAMES = ("train", "val", "calib", "test_internal")


def _patient_labels(frame: pd.DataFrame) -> pd.DataFrame:
    patient_frame = (
        frame.groupby("pid", as_index=False)["label_cancer_binary"]
        .max()
        .rename(columns={"label_cancer_binary": "patient_cancer"})
    )
    patient_frame["pid"] = patient_frame["pid"].astype(str)
    return patient_frame


def _split_patients(
    patient_frame: pd.DataFrame,
    *,
    seed: int,
    ratios: tuple[float, float, float, float],
) -> dict[str, set[str]]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        msg = f"Split ratios must sum to 1.0, got {ratios}"
        raise ValueError(msg)

    pids = patient_frame["pid"].astype(str)
    labels = patient_frame["patient_cancer"].astype(int)
    train_ratio, val_ratio, calib_ratio, test_ratio = ratios

    train_pids, remaining_pids, _train_y, remaining_y = train_test_split(
        pids,
        labels,
        test_size=1.0 - train_ratio,
        random_state=seed,
        stratify=labels,
    )
    val_fraction_of_remaining = val_ratio / (val_ratio + calib_ratio + test_ratio)
    val_pids, holdout_pids, _val_y, holdout_y = train_test_split(
        remaining_pids,
        remaining_y,
        test_size=1.0 - val_fraction_of_remaining,
        random_state=seed + 1,
        stratify=remaining_y,
    )
    calib_fraction_of_holdout = calib_ratio / (calib_ratio + test_ratio)
    calib_pids, test_pids, _calib_y, _test_y = train_test_split(
        holdout_pids,
        holdout_y,
        test_size=1.0 - calib_fraction_of_holdout,
        random_state=seed + 2,
        stratify=holdout_y,
    )

    return {
        "train": set(train_pids.astype(str)),
        "val": set(val_pids.astype(str)),
        "calib": set(calib_pids.astype(str)),
        "test_internal": set(test_pids.astype(str)),
    }


def assign_patient_splits(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float, float] = (0.60, 0.15, 0.10, 0.15),
) -> pd.DataFrame:
    working = frame.copy()
    working["pid"] = working["pid"].astype(str)
    patient_splits = _split_patients(_patient_labels(working), seed=seed, ratios=ratios)
    split_by_pid = {
        pid: split_name
        for split_name, pids in patient_splits.items()
        for pid in pids
    }
    working["split_v2"] = working["pid"].map(split_by_pid)
    if working["split_v2"].isna().any():
        msg = "Some CystoDS rows were not assigned to a v2 split."
        raise ValueError(msg)
    assert_no_patient_leakage_v2(working)
    assert_each_split_has_both_classes(working)
    return working


def assert_no_patient_leakage_v2(frame: pd.DataFrame, *, split_column: str = "split_v2") -> None:
    counts = frame.groupby("pid")[split_column].nunique()
    leaked = counts[counts > 1]
    if not leaked.empty:
        msg = f"Patient leakage detected across v2 splits: {leaked.index.tolist()[:10]}"
        raise AssertionError(msg)


def assert_each_split_has_both_classes(frame: pd.DataFrame, *, split_column: str = "split_v2") -> None:
    for split_name in V2_SPLIT_NAMES:
        subset = frame[frame[split_column] == split_name]
        classes = set(subset["label_cancer_binary"].dropna().astype(int))
        if classes != {0, 1}:
            msg = f"Split {split_name} must contain cancer=0 and cancer=1, got {sorted(classes)}"
            raise AssertionError(msg)


def write_split_audit(
    cystods: pd.DataFrame,
    ebtc: pd.DataFrame,
    output_path: str | Path,
) -> None:
    lines = ["# Stage 02 Split Audit", ""]
    lines.extend(["## CystoDS v2 Splits", ""])
    rows = ["| split | rows | patients | cancer_positive | cancer_negative |", "| --- | --- | --- | --- | --- |"]
    for split_name in V2_SPLIT_NAMES:
        subset = cystods[cystods["split_v2"] == split_name]
        positives = int((subset["label_cancer_binary"].astype(int) == 1).sum())
        negatives = int((subset["label_cancer_binary"].astype(int) == 0).sum())
        rows.append(
            f"| {split_name} | {len(subset)} | {subset['pid'].nunique()} | {positives} | {negatives} |"
        )
    lines.append("\n".join(rows))
    lines.extend(["", "## EBTC External Splits", ""])
    rows = ["| split | rows | patients |", "| --- | --- | --- |"]
    rows.append(f"| external_all | {len(ebtc)} | {ebtc['pid'].nunique()} |")
    official = ebtc[ebtc["split_source"].astype(str).str.lower() == "test"]
    rows.append(f"| external_official_test | {len(official)} | {official['pid'].nunique()} |")
    lines.append("\n".join(rows))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_v2_splits(
    cystods_metadata: str | Path,
    ebtc_metadata: str | Path,
    *,
    output_dir: str | Path = "outputs/v2/splits",
    seed: int = 42,
    ratios: tuple[float, float, float, float] = (0.60, 0.15, 0.10, 0.15),
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cystods = pd.read_csv(cystods_metadata)
    ebtc = pd.read_csv(ebtc_metadata)
    cystods_split = assign_patient_splits(cystods, seed=seed, ratios=ratios)

    split_paths: dict[str, Path] = {}
    for split_name in V2_SPLIT_NAMES:
        path = output / f"cystods_{split_name}.csv"
        write_table(cystods_split[cystods_split["split_v2"] == split_name], path)
        split_paths[f"cystods_{split_name}"] = path

    external_all = output / "ebtc_external_all.csv"
    official_test = output / "ebtc_external_official_test.csv"
    write_table(ebtc, external_all)
    write_table(ebtc[ebtc["split_source"].astype(str).str.lower() == "test"], official_test)
    split_paths["ebtc_external_all"] = external_all
    split_paths["ebtc_external_official_test"] = official_test

    split_manifest = output / "split_manifest.json"
    write_split_manifest(split_paths, split_manifest)
    audit_path = output / "stage02_split_audit.md"
    write_split_audit(cystods_split, ebtc, audit_path)
    return {
        "cystods": cystods_split,
        "ebtc": ebtc,
        "split_paths": split_paths,
        "split_manifest": split_manifest,
        "audit_path": audit_path,
    }
