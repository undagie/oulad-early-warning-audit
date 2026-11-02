"""Fitur engagement temporal mingguan dari aktivitas VLE.

Window "Week k" = k minggu pertama (indeks minggu 0..k-1), sehingga sekuens punya
K = k langkah. Aktivitas pra-mulai (date < 0) diklип ke Week 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nsoulad.data.schema import UNIT_KEYS
from nsoulad.data.windowing import WindowView

# Tipe aktivitas yang disebut eksplisit di desain.
NAMED_ACTIVITY_TYPES = ["resource", "forumng", "homepage", "quiz", "subpage"]

# Urutan fitur per (mahasiswa, minggu) pada tensor sekuens.
SEQ_FEATURES = [
    "clicks",
    "active_days",
    "mean_clicks_active",
    "n_activity_types",
    *[f"clicks_{a}" for a in NAMED_ACTIVITY_TYPES],
    "is_inactive",
    "delta_clicks",
    "prop_cumulative",
]

_KEYS = list(UNIT_KEYS)


def weekly_engagement(view: WindowView, days_per_week: int = 7) -> pd.DataFrame:
    """Tabel tidy fitur per (mahasiswa, minggu) untuk seluruh cohort window.

    Grid penuh cohort x {0..K-1} dipastikan terisi (minggu tanpa aktivitas -> 0).
    """
    K = view.k
    svle = view.student_vle.merge(
        view.vle[["id_site", "code_module", "code_presentation", "activity_type"]],
        on=["id_site", "code_module", "code_presentation"],
        how="left",
    ).copy()

    if len(svle):
        date_f = svle["date"].astype("float")
        week = np.floor(date_f / days_per_week).clip(lower=0).astype(int)
        svle["week"] = week
        svle = svle[svle["week"] < K]  # guard: hanya minggu dalam window

    # Agregat dasar per (mahasiswa, minggu).
    if len(svle):
        agg = (
            svle.groupby(_KEYS + ["week"])
            .agg(
                clicks=("sum_click", "sum"),
                active_days=("date", "nunique"),
                n_activity_types=("activity_type", "nunique"),
            )
            .reset_index()
        )
        for at in NAMED_ACTIVITY_TYPES:
            sub = (
                svle[svle["activity_type"] == at]
                .groupby(_KEYS + ["week"])["sum_click"]
                .sum()
                .reset_index(name=f"clicks_{at}")
            )
            agg = agg.merge(sub, on=_KEYS + ["week"], how="left")
    else:
        cols = _KEYS + ["week", "clicks", "active_days", "n_activity_types"] + [
            f"clicks_{a}" for a in NAMED_ACTIVITY_TYPES
        ]
        agg = pd.DataFrame(columns=cols)

    # Grid penuh: setiap mahasiswa cohort x setiap minggu 0..K-1.
    cohort = view.cohort_keys[_KEYS].drop_duplicates().reset_index(drop=True)
    weeks = pd.DataFrame({"week": range(K)})
    grid = cohort.merge(weeks, how="cross")
    full = grid.merge(agg, on=_KEYS + ["week"], how="left")

    base_cols = ["clicks", "active_days", "n_activity_types"] + [
        f"clicks_{a}" for a in NAMED_ACTIVITY_TYPES
    ]
    full[base_cols] = full[base_cols].fillna(0.0).astype(float)

    # Fitur turunan.
    full["mean_clicks_active"] = np.where(
        full["active_days"] > 0, full["clicks"] / full["active_days"], 0.0
    )
    full["is_inactive"] = (full["clicks"] == 0).astype(float)
    full = full.sort_values(_KEYS + ["week"]).reset_index(drop=True)
    full["delta_clicks"] = full.groupby(_KEYS)["clicks"].diff().fillna(0.0)
    cum = full.groupby(_KEYS)["clicks"].cumsum()
    full["prop_cumulative"] = np.where(cum > 0, full["clicks"] / cum, 0.0)
    return full


def to_sequence_tensor(
    weekly_df: pd.DataFrame, cohort_keys: pd.DataFrame, K: int
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Konversi tabel tidy mingguan -> tensor [n, K, F] selaras urutan cohort_keys.

    Returns (tensor, feature_names, order_keys) dengan order_keys berisi UNIT_KEYS
    pada urutan baris tensor.
    """
    order = cohort_keys[_KEYS].drop_duplicates().reset_index(drop=True)
    order = order.assign(__i=range(len(order)))
    df = weekly_df.merge(order, on=_KEYS, how="left").sort_values(["__i", "week"])

    n, F = len(order), len(SEQ_FEATURES)
    if len(df) != n * K:
        raise ValueError(
            f"Grid mingguan tidak konsisten: {len(df)} baris, diharapkan {n * K} (n={n}, K={K})."
        )
    tensor = df[SEQ_FEATURES].to_numpy(dtype=float).reshape(n, K, F)
    return tensor, list(SEQ_FEATURES), order.drop(columns="__i")
