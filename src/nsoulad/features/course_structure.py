"""Fitur course-structure: keterhubungan aktivitas dengan struktur & deadline.

Menangkap apakah mahasiswa mengakses resource terjadwal dan aktif menjelang
deadline assessment. Semua dari WindowView (bebas-bocor).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nsoulad.data.schema import UNIT_KEYS
from nsoulad.data.windowing import WindowView

_KEYS = list(UNIT_KEYS)
_COURSE = ["code_module", "code_presentation"]

COURSE_STRUCTURE_FEATURES = [
    "scheduled_resource_ratio",
    "accessed_before_any_deadline",
    "active_days_7d_pre_deadline",
]


def course_structure_features(view: WindowView, days_per_week: int = 7) -> pd.DataFrame:
    """Fitur course-structure per mahasiswa cohort (reindex ke seluruh cohort)."""
    cohort = view.cohort_keys[_KEYS].drop_duplicates().reset_index(drop=True)
    K = view.k
    cur_week = K - 1  # minggu terkini dalam window

    svle = view.student_vle
    vle = view.vle

    # --- 1) Rasio scheduled resources yang diakses ---
    # "Scheduled" = site dengan week_from eksplisit (terjadwal ke minggu tertentu) yang
    # sudah tersedia pada window (week_from <= cur_week). Site tanpa jadwal (week_from NaN,
    # ~82% di OULAD) bukan "scheduled resource" sehingga tidak dihitung di sini.
    avail = vle[vle["week_from"].notna() & (vle["week_from"] <= cur_week)]
    n_avail = (
        avail.groupby(_COURSE)["id_site"].nunique().reset_index(name="n_avail_sites")
    )
    if len(svle):
        accessed = (
            svle.merge(avail[_COURSE + ["id_site"]].drop_duplicates(),
                       on=_COURSE + ["id_site"], how="inner")
            .groupby(_KEYS)["id_site"].nunique().reset_index(name="n_accessed_sites")
        )
    else:
        accessed = pd.DataFrame(columns=_KEYS + ["n_accessed_sites"])

    out = cohort.merge(accessed, on=_KEYS, how="left").merge(n_avail, on=_COURSE, how="left")
    out["n_accessed_sites"] = out["n_accessed_sites"].fillna(0.0)
    out["n_avail_sites"] = out["n_avail_sites"].fillna(0.0)
    # Bila tak ada scheduled resource (n_avail==0), ratio=1.0 (tidak ada yg "terlewat").
    out["scheduled_resource_ratio"] = np.where(
        out["n_avail_sites"] > 0, out["n_accessed_sites"] / out["n_avail_sites"], 1.0
    )

    # --- 2) Aktivitas menjelang deadline ---
    due = view.assessments_due
    if len(svle) and len(due):
        # Untuk tiap submission-window deadline d: hitung hari aktif pada [d-7, d-1].
        sv = svle[_KEYS + ["date"]].copy()
        sv["date"] = sv["date"].astype("float")
        deadlines = due[_COURSE + ["date"]].dropna(subset=["date"]).drop_duplicates()
        deadlines = deadlines.rename(columns={"date": "deadline"})
        # Join aktivitas mahasiswa dengan deadline course-nya.
        joined = sv.merge(deadlines, on=_COURSE, how="inner")
        pre = joined[(joined["date"] >= joined["deadline"] - days_per_week) & (joined["date"] < joined["deadline"])]
        pre_stats = (
            pre.groupby(_KEYS)
            .agg(active_days_7d_pre_deadline=("date", "nunique"))
            .reset_index()
        )
        pre_stats["accessed_before_any_deadline"] = (pre_stats["active_days_7d_pre_deadline"] > 0).astype(float)
    else:
        pre_stats = pd.DataFrame(columns=_KEYS + ["active_days_7d_pre_deadline", "accessed_before_any_deadline"])

    out = out.merge(pre_stats, on=_KEYS, how="left")
    out["active_days_7d_pre_deadline"] = out["active_days_7d_pre_deadline"].fillna(0.0)
    out["accessed_before_any_deadline"] = out["accessed_before_any_deadline"].fillna(0.0)

    return out[_KEYS + COURSE_STRUCTURE_FEATURES]
