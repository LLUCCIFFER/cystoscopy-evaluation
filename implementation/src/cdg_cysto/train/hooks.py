"""Training hooks and small state trackers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BestMetricTracker:
    metric_name: str = "macro_f1"
    mode: str = "max"
    best_value: float | None = None
    best_epoch: int | None = None

    def update(self, epoch: int, metrics: dict[str, float]) -> bool:
        value = metrics.get(self.metric_name)
        if value is None:
            return False
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            return True
        improved = value > self.best_value if self.mode == "max" else value < self.best_value
        if improved:
            self.best_value = value
            self.best_epoch = epoch
        return improved


@dataclass
class EarlyStopping:
    patience: int = 10
    mode: str = "max"
    best_value: float | None = None
    num_bad_epochs: int = 0

    def step(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            self.num_bad_epochs = 0
            return False
        improved = value > self.best_value if self.mode == "max" else value < self.best_value
        if improved:
            self.best_value = value
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience
