"""Util model: perakitan fitur datar & alignment output symbolic ke baris dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nsoulad.data.schema import UNIT_KEYS
from nsoulad.features.build import WindowDataset, flatten_sequence

_KEYS = list(UNIT_KEYS)


def full_features(ds: WindowDataset) -> pd.DataFrame:
    """Fitur datar penuh: ringkasan sekuens temporal + tabular (static+assess+course)."""
    flat = flatten_sequence(ds.X_seq, ds.seq_features)
    return pd.concat([flat.reset_index(drop=True), ds.X_tab.reset_index(drop=True)], axis=1)


def static_features(ds: WindowDataset) -> pd.DataFrame:
    """Hanya fitur demografi statis (kontrol static-feature dominance)."""
    return ds.X_tab[ds.encoder.feature_names_].reset_index(drop=True)


@dataclass
class SymbolicBundle:
    """Output symbolic selaras urutan baris satu WindowDataset."""

    p_symbolic: np.ndarray
    rule_count: np.ndarray
    high_severity_count: np.ndarray
    triggered: pd.DataFrame  # [n,7]

    def fusion_matrix(self, p_neural: np.ndarray) -> np.ndarray:
        """Matriks fitur fusion: [P_neural, P_symbolic, rule_count, high_severity_count]."""
        return np.column_stack([p_neural, self.p_symbolic, self.rule_count, self.high_severity_count])

    def as_features(self) -> np.ndarray:
        """Fitur symbolic untuk rule-augmented GRU: [triggered(7) | p_symbolic | rule_count | high_sev]."""
        return np.column_stack([
            self.triggered.to_numpy(dtype=float),
            self.p_symbolic, self.rule_count, self.high_severity_count,
        ])


def align_symbolic(sym_frame: pd.DataFrame, keys: pd.DataFrame) -> SymbolicBundle:
    """Selaraskan RuleResult.to_frame() ke urutan baris `keys` (UNIT_KEYS)."""
    from nsoulad.symbolic.rules import RULE_IDS

    merged = keys.merge(sym_frame, on=_KEYS, how="left")
    if merged["p_symbolic"].isna().any():
        raise ValueError("Ada baris tanpa output symbolic saat alignment.")
    return SymbolicBundle(
        p_symbolic=merged["p_symbolic"].to_numpy(float),
        rule_count=merged["rule_count"].to_numpy(float),
        high_severity_count=merged["high_severity_count"].to_numpy(float),
        triggered=merged[RULE_IDS].reset_index(drop=True),
    )
