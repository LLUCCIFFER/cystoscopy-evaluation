"""Configuration helpers for CDG-Cysto experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - exercised only in minimal environments
    OmegaConf = None  # type: ignore[assignment]

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


EXPECTED_RUN_ARTIFACTS = [
    "config_resolved.yaml",
    "metrics_epoch.csv",
    "metrics_val_best.json",
    "metrics_test.json",
    "predictions_test.csv",
    "confusion_matrix.png",
    "roc_curve.png",
    "pr_curve.png",
    "reliability_diagram.png",
    "checkpoint_best.pt",
    "run.log",
]


@dataclass(frozen=True)
class DatasetColumns:
    image: str = "image_path"
    label: str = "label"
    patient: str = "patient_id"
    split: str = "split"
    domain: str | None = None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    metadata_csv: Path
    classes: list[str]
    columns: DatasetColumns = field(default_factory=DatasetColumns)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_classes: int
    input_size: int = 224
    backbone: str = "dinov2_vits14"
    feature_dim: int = 384
    pretrained: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    epochs_stage1: int = 30
    epochs_stage2: int = 20
    batch_size: int = 16
    learning_rate_head: float = 3e-4
    learning_rate_backbone: float = 1e-5


def require_omegaconf() -> Any:
    if OmegaConf is None:
        msg = "omegaconf is required for configuration loading. Install requirements.txt first."
        raise ImportError(msg)
    return OmegaConf


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config as a plain Python dictionary."""

    path = Path(path)
    if OmegaConf is not None:
        cfg = OmegaConf.load(path)
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[no-any-return]
    if yaml is not None:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)  # type: ignore[no-any-return]
    msg = "Either omegaconf or pyyaml is required for configuration loading."
    raise ImportError(msg)


def merge_yaml_files(*paths: str | Path) -> dict[str, Any]:
    """Merge YAML files from left to right and return a resolved dictionary."""

    omega_conf = require_omegaconf()
    configs = [omega_conf.load(Path(path)) for path in paths]
    merged = omega_conf.merge(*configs)
    return omega_conf.to_container(merged, resolve=True)  # type: ignore[no-any-return]


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a resolved config using OmegaConf when available."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if OmegaConf is not None:
        OmegaConf.save(OmegaConf.create(config), target)
        return
    if yaml is not None:
        with target.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
        return
    msg = "Either omegaconf or pyyaml is required for configuration saving."
    raise ImportError(msg)


def resolve_experiment_config(config_path: str | Path) -> dict[str, Any]:
    """Load an experiment config and inline referenced dataset/model configs."""

    config_path = Path(config_path)
    experiment = load_yaml(config_path)
    base_dir = config_path.parent.parent.parent

    for key in ("dataset_config", "external_dataset_config", "model_config"):
        if key in experiment:
            referenced = base_dir / str(experiment[key])
            experiment[key.replace("_config", "")] = load_yaml(referenced)

    if "models" in experiment:
        experiment["resolved_models"] = [load_yaml(base_dir / str(path)) for path in experiment["models"]]

    experiment["config_path"] = str(config_path)
    return experiment


def expected_artifact_paths(run_dir: str | Path) -> dict[str, Path]:
    run_path = Path(run_dir)
    return {name: run_path / name for name in EXPECTED_RUN_ARTIFACTS}
