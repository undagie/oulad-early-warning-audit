"""Encoder fitur statis (demografi) — fit-on-train, leak-free.

Ordinal: highest_education, age_band, imd_band (di-standardisasi).
One-hot: gender, region, disability, code_module, code_presentation.
Numerik: num_of_prev_attempts, studied_credits (di-standardisasi).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

ORDINAL_ORDERS: dict[str, list[str]] = {
    "highest_education": [
        "No Formal quals",
        "Lower Than A Level",
        "A Level or Equivalent",
        "HE Qualification",
        "Post Graduate Qualification",
    ],
    "age_band": ["0-35", "35-55", "55<="],
}
ONEHOT_COLS = ["gender", "region", "disability", "code_module", "code_presentation"]
NUMERIC_COLS = ["num_of_prev_attempts", "studied_credits"]
# Kolom ordinal yang akan di-standardisasi bersama numerik.
_ORDINAL_NUMERIC = list(ORDINAL_ORDERS) + ["imd_band"] + NUMERIC_COLS


def _imd_to_int(val) -> float:
    """imd_band -> batas bawah (mis. '20-30%' -> 20). NaN/aneh -> NaN."""
    if pd.isna(val):
        return np.nan
    m = re.match(r"\s*(\d+)", str(val))
    return float(m.group(1)) if m else np.nan


def _to_ordinal(series: pd.Series, order: list[str]) -> pd.Series:
    mapping = {cat: i for i, cat in enumerate(order)}
    return series.map(mapping).astype("float")


class StaticEncoder:
    """Encoder demografi: fit pada train, transform pada split apa pun."""

    def __init__(self) -> None:
        self.onehot_categories_: dict[str, list] = {}
        self.medians_: dict[str, float] = {}
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None
        self.feature_names_: list[str] = []
        self._fitted = False

    def _ordinal_block(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col, order in ORDINAL_ORDERS.items():
            out[col] = _to_ordinal(df[col], order)
        out["imd_band"] = df["imd_band"].map(_imd_to_int)
        for col in NUMERIC_COLS:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        return out[_ORDINAL_NUMERIC]

    def fit(self, df: pd.DataFrame) -> "StaticEncoder":
        for col in ONEHOT_COLS:
            cats = sorted(df[col].dropna().astype(str).unique().tolist())
            self.onehot_categories_[col] = cats

        block = self._ordinal_block(df)
        # Median train per kolom; bila seluruh kolom NaN (kategori tak dikenal), pakai 0.0.
        self.medians_ = {
            c: (0.0 if np.isnan(block[c].median()) else float(block[c].median()))
            for c in block.columns
        }
        block_imp = block.fillna(self.medians_)
        self.mean_ = block_imp.mean()
        self.std_ = block_imp.std(ddof=0).replace(0.0, 1.0)

        # Tetapkan urutan kolom keluaran.
        names = list(_ORDINAL_NUMERIC)
        for col in ONEHOT_COLS:
            names += [f"{col}={c}" for c in self.onehot_categories_[col]]
        self.feature_names_ = names
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("StaticEncoder belum di-fit.")
        block = self._ordinal_block(df).fillna(self.medians_).fillna(0.0)
        block = (block - self.mean_) / self.std_

        onehot = {}
        for col in ONEHOT_COLS:
            vals = df[col].astype(str)
            for cat in self.onehot_categories_[col]:
                onehot[f"{col}={cat}"] = (vals == cat).astype(float).to_numpy()
        onehot_df = pd.DataFrame(onehot, index=df.index)

        out = pd.concat([block.reset_index(drop=True), onehot_df.reset_index(drop=True)], axis=1)
        return out[self.feature_names_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
