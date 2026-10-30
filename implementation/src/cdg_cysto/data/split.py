"""Split helpers with patient-level leakage checks."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from cdg_cysto.utils.io import write_table


def patient_sets_by_split(
    frame: pd.DataFrame,
    split_column: str = "split",
    patient_column: str = "patient_id",
) -> dict[str, set[str]]:
    missing = {split_column, patient_column} - set(frame.columns)
    if missing:
        msg = f"Missing split columns: {sorted(missing)}"
        raise ValueError(msg)

    result: dict[str, set[str]] = {}
    for split_name, group in frame.groupby(split_column):
        result[str(split_name)] = {str(value) for value in group[patient_column].dropna().unique()}
    return result


def find_patient_leakage(
    frame: pd.DataFrame,
    split_column: str = "split",
    patient_column: str = "patient_id",
) -> dict[tuple[str, str], set[str]]:
    split_sets = patient_sets_by_split(frame, split_column, patient_column)
    leakage: dict[tuple[str, str], set[str]] = {}
    for left, right in combinations(sorted(split_sets), 2):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            leakage[(left, right)] = overlap
    return leakage


def assert_no_patient_leakage(
    frame: pd.DataFrame,
    split_column: str = "split",
    patient_column: str = "patient_id",
) -> None:
    leakage = find_patient_leakage(frame, split_column, patient_column)
    if leakage:
        details = ", ".join(
            f"{left}/{right}: {sorted(patients)}" for (left, right), patients in leakage.items()
        )
        msg = f"Patient leakage detected across splits: {details}"
        raise ValueError(msg)


def make_patient_level_split(
    frame: pd.DataFrame,
    *,
    patient_column: str = "pid",
    label_column: str = "label_original",
    seed: int = 42,
    train_size: float = 0.70,
    val_size: float = 0.15,
) -> pd.DataFrame:
    """Assign train/val/test splits at patient level.

    Stratification uses each patient's most frequent label when every stratum has
    at least two patients; otherwise it falls back to deterministic unstratified
    splitting.
    """

    missing = {patient_column, label_column} - set(frame.columns)
    if missing:
        msg = f"Missing patient split columns: {sorted(missing)}"
        raise ValueError(msg)
    if not 0 < train_size < 1 or not 0 <= val_size < 1 or train_size + val_size >= 1:
        msg = "train_size and val_size must leave a positive test fraction."
        raise ValueError(msg)

    patient_labels = (
        frame[[patient_column, label_column]]
        .dropna(subset=[patient_column])
        .assign(**{patient_column: lambda x: x[patient_column].astype(str)})
        .groupby(patient_column)[label_column]
        .agg(lambda values: values.astype(str).value_counts().idxmax())
        .reset_index()
        .rename(columns={patient_column: "pid", label_column: "stratum"})
    )
    patients = patient_labels["pid"].tolist()
    strata = patient_labels["stratum"].tolist()
    stratify = strata if pd.Series(strata).value_counts().min() >= 2 else None

    train_patients, remaining_patients, _, remaining_strata = train_test_split(
        patients,
        strata,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    remaining_val_fraction = val_size / (1.0 - train_size)
    remaining_counts = pd.Series(remaining_strata).value_counts()
    remaining_classes = len(remaining_counts)
    expected_val = int(round(len(remaining_patients) * remaining_val_fraction))
    expected_test = len(remaining_patients) - expected_val
    remaining_stratify = (
        remaining_strata
        if remaining_counts.min() >= 2
        and len(remaining_patients) > 2 * remaining_classes
        and expected_val >= remaining_classes
        and expected_test >= remaining_classes
        else None
    )
    val_patients, test_patients = train_test_split(
        remaining_patients,
        train_size=remaining_val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=remaining_stratify,
    )

    rows = [
        *({"pid": pid, "split": "train"} for pid in train_patients),
        *({"pid": pid, "split": "val"} for pid in val_patients),
        *({"pid": pid, "split": "test"} for pid in test_patients),
    ]
    split_frame = pd.DataFrame(rows).sort_values(["split", "pid"]).reset_index(drop=True)
    assert_no_patient_leakage(split_frame, patient_column="pid")
    return split_frame


def apply_patient_split(
    frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    *,
    patient_column: str = "pid",
) -> pd.DataFrame:
    """Attach a patient-level split table to metadata."""

    mapping = dict(zip(split_frame["pid"].astype(str), split_frame["split"], strict=False))
    output = frame.copy()
    output["split"] = output[patient_column].astype(str).map(mapping).fillna("external")
    assert_no_patient_leakage(output, patient_column=patient_column)
    return output


def write_split_membership(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    split_column: str = "split",
    sample_column: str = "sample_id",
) -> None:
    """Write one CSV per split containing sample IDs."""

    target_dir = Path(output_dir)
    for split_name, group in frame.groupby(split_column):
        write_table(group[[sample_column]], target_dir / f"{split_name}.csv")


def write_ebtc_domain_splits(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    modality_column: str = "modality",
    split_column: str = "split",
) -> None:
    """Write WLI/NBI within-domain and cross-domain split manifests."""

    target_dir = Path(output_dir)
    for modality in ("WLI", "NBI"):
        domain = frame[frame[modality_column].astype(str).str.upper() == modality]
        for split_name in ("train", "val", "test"):
            subset = domain[domain[split_column].astype(str).str.lower() == split_name]
            write_table(subset[["sample_id"]], target_dir / f"{modality.lower()}_{split_name}.csv")

    pairs = (("WLI", "NBI"), ("NBI", "WLI"))
    for source, target in pairs:
        train = frame[
            (frame[modality_column].astype(str).str.upper() == source)
            & (frame[split_column].astype(str).str.lower() == "train")
        ]
        test = frame[frame[modality_column].astype(str).str.upper() == target]
        write_table(train[["sample_id"]], target_dir / f"{source.lower()}_to_{target.lower()}_train.csv")
        write_table(test[["sample_id"]], target_dir / f"{source.lower()}_to_{target.lower()}_test.csv")


def filter_split(
    frame: pd.DataFrame,
    split: str | None = None,
    domain: str | None = None,
    split_column: str = "split",
    domain_column: str = "imaging_mode",
) -> pd.DataFrame:
    filtered = frame
    if split is not None:
        filtered = filtered[filtered[split_column].astype(str).str.lower() == split.lower()]
    if domain is not None and domain_column in filtered.columns:
        filtered = filtered[filtered[domain_column].astype(str).str.upper() == domain.upper()]
    return filtered.reset_index(drop=True)


def summarize_splits(
    frame: pd.DataFrame,
    split_column: str = "split",
    label_column: str = "label",
    patient_column: str = "patient_id",
) -> pd.DataFrame:
    grouped = frame.groupby([split_column, label_column], dropna=False)
    summary = grouped.agg(images=(label_column, "size"), patients=(patient_column, "nunique"))
    return summary.reset_index()
