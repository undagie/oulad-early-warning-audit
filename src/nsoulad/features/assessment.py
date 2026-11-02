"""Fitur asesmen dari submission & deadline yang sudah jatuh pada window.

Semua input berasal dari WindowView (sudah dipotong cutoff-first):
  - view.assessments_due: assessment dengan deadline <= cutoff;
  - view.student_assessment: submission <= cutoff untuk assessment due tsb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nsoulad.data.schema import UNIT_KEYS
from nsoulad.data.windowing import WindowView

_KEYS = list(UNIT_KEYS)

ASSESSMENT_FEATURES = [
    "n_submitted",
    "first_score",
    "mean_score",
    "n_late",
    "mean_lateness",
    "completion_ratio",
    "total_weight_done",
    "has_assessment",
]

# Kolom bantu (bukan fitur model): jumlah submission yang skornya sudah terlihat, dan
# penanda "sudah submit tetapi belum ada skor terlihat" (skor NaN di OULAD atau tertahan
# oleh score_lag_days). Baris score_pending diimputasi dengan rata-rata train oleh pipeline
# revisi, bukan dengan 0, agar submission yang belum dinilai tidak terbaca sebagai skor 0.
AUX_ASSESSMENT_COLUMNS = ["n_scored", "score_pending"]


def assessment_features(view: WindowView) -> pd.DataFrame:
    """Fitur asesmen per mahasiswa cohort (di-reindex ke seluruh cohort).

    Tugas yang belum dikumpulkan sampai cutoff masuk lewat completion_ratio (pembilang
    tidak bertambah) dan has_assessment; skor mahasiswa tanpa submission diisi 0 dengan
    has_assessment = 0 sebagai penanda.
    """
    cohort = view.cohort_keys[_KEYS].drop_duplicates().reset_index(drop=True)

    due = view.assessments_due
    sa = view.student_assessment

    # Jumlah assessment due per course (penyebut completion ratio).
    if len(due):
        n_due = (
            due.groupby(["code_module", "code_presentation"])["id_assessment"]
            .nunique()
            .reset_index(name="n_due")
        )
    else:
        n_due = pd.DataFrame(columns=["code_module", "code_presentation", "n_due"])

    if len(sa):
        # Lengkapi submission dengan metadata assessment (deadline, weight, course).
        merged = sa.merge(
            due[["id_assessment", "code_module", "code_presentation", "date", "weight"]],
            on="id_assessment",
            how="inner",
        ).copy()
        merged["lateness"] = (merged["date_submitted"].astype("float") - merged["date"].astype("float")).clip(lower=0)
        merged["is_late"] = (merged["date_submitted"].astype("float") > merged["date"].astype("float")).astype(int)

        # first_score: skor terlihat pada deadline paling awal (lalu submit paling awal).
        scored = merged[merged["score"].notna()]
        order = scored.sort_values(_KEYS + ["date", "date_submitted"])
        first = order.groupby(_KEYS).first().reset_index()[_KEYS + ["score"]]
        first = first.rename(columns={"score": "first_score"})

        grp = merged.groupby(_KEYS)
        feats = grp.agg(
            n_submitted=("id_assessment", "nunique"),
            n_scored=("score", "count"),
            mean_score=("score", "mean"),
            n_late=("is_late", "sum"),
            mean_lateness=("lateness", "mean"),
            total_weight_done=("weight", "sum"),
        ).reset_index()
        feats = feats.merge(first, on=_KEYS, how="left")
    else:
        feats = pd.DataFrame(columns=_KEYS + [
            "n_submitted", "n_scored", "mean_score", "n_late", "mean_lateness",
            "total_weight_done", "first_score"
        ])

    out = cohort.merge(feats, on=_KEYS, how="left")
    out = out.merge(n_due, on=["code_module", "code_presentation"], how="left")

    out["has_assessment"] = out["n_submitted"].notna().astype(float)
    for col in ["n_submitted", "n_scored", "mean_score", "n_late", "mean_lateness",
                "total_weight_done", "first_score"]:
        out[col] = out[col].fillna(0.0).astype(float)
    out["score_pending"] = ((out["n_submitted"] > 0) & (out["n_scored"] == 0)).astype(float)
    out["n_due"] = out["n_due"].fillna(0.0)
    out["completion_ratio"] = np.where(out["n_due"] > 0, out["n_submitted"] / out["n_due"], 0.0)

    return out[_KEYS + ASSESSMENT_FEATURES + AUX_ASSESSMENT_COLUMNS]
