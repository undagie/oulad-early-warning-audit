"""Konfigurasi eksperimen — dibaca dari config/config.yaml dengan default aman."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SplitConfig:
    scheme: str = "student"  # "student" | "presentation"
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    train_presentations: tuple[str, ...] = ("2013B", "2013J")
    test_presentations: tuple[str, ...] = ("2014B", "2014J")


@dataclass(frozen=True)
class LabelConfig:
    at_risk: tuple[str, ...] = ("Fail", "Withdrawn")
    not_at_risk: tuple[str, ...] = ("Pass", "Distinction")


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data/raw")
    windows: tuple[int, ...] = (2, 4, 6, 8)
    days_per_week: int = 7
    seed: int = 42
    split: SplitConfig = field(default_factory=SplitConfig)
    label: LabelConfig = field(default_factory=LabelConfig)


def load_config(path: str | Path = "config/config.yaml") -> Config:
    """Muat konfigurasi dari YAML; jika file tidak ada, pakai default."""
    p = Path(path)
    if not p.exists():
        return Config()

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    split_raw = raw.get("split", {}) or {}
    label_raw = raw.get("label", {}) or {}

    return Config(
        data_dir=Path(raw.get("data_dir", "data/raw")),
        windows=tuple(raw.get("windows", (2, 4, 6, 8))),
        days_per_week=int(raw.get("days_per_week", 7)),
        seed=int(raw.get("seed", 42)),
        split=SplitConfig(
            scheme=split_raw.get("scheme", "student"),
            ratios=tuple(split_raw.get("ratios", (0.70, 0.15, 0.15))),
            train_presentations=tuple(split_raw.get("train_presentations", ("2013B", "2013J"))),
            test_presentations=tuple(split_raw.get("test_presentations", ("2014B", "2014J"))),
        ),
        label=LabelConfig(
            at_risk=tuple(label_raw.get("at_risk", ("Fail", "Withdrawn"))),
            not_at_risk=tuple(label_raw.get("not_at_risk", ("Pass", "Distinction"))),
        ),
    )
