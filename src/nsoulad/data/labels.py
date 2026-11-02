"""Konstruksi label biner at-risk.

Anti-leakage: `final_result` HANYA dipakai sebagai label, tidak pernah sebagai fitur.
"""

from __future__ import annotations

import pandas as pd

from nsoulad.data.schema import FINAL_RESULT_VALUES, UNIT_KEYS

DEFAULT_AT_RISK = ("Fail", "Withdrawn")
DEFAULT_NOT_AT_RISK = ("Pass", "Distinction")


def make_binary_label(
    student_info: pd.DataFrame,
    at_risk: tuple[str, ...] = DEFAULT_AT_RISK,
    not_at_risk: tuple[str, ...] = DEFAULT_NOT_AT_RISK,
) -> pd.DataFrame:
    """Bangun label biner per unit analisis (mahasiswa x module-presentation).

    Returns kolom: UNIT_KEYS + [final_result, at_risk] dengan
    at_risk = 1 untuk Fail/Withdrawn, 0 untuk Pass/Distinction.

    Raises ValueError bila ada nilai final_result yang tidak dikenal, atau bila
    pemetaan at_risk/not_at_risk tidak menutup seluruh kelas final_result.
    """
    if "final_result" not in student_info.columns:
        raise ValueError("student_info tidak memiliki kolom 'final_result'.")

    mapping = {v: 1 for v in at_risk}
    mapping.update({v: 0 for v in not_at_risk})

    uncovered = FINAL_RESULT_VALUES - set(mapping)
    if uncovered:
        raise ValueError(
            f"Pemetaan label tidak menutup kelas final_result: {sorted(uncovered)}. "
            f"Periksa konfigurasi label."
        )

    unknown = set(student_info["final_result"].dropna().unique()) - FINAL_RESULT_VALUES
    if unknown:
        raise ValueError(f"Nilai final_result tak dikenal pada data: {sorted(unknown)}")

    out = student_info[list(UNIT_KEYS) + ["final_result"]].copy()
    out["at_risk"] = out["final_result"].map(mapping).astype("Int64")
    if out["at_risk"].isna().any():
        bad = out.loc[out["at_risk"].isna(), "final_result"].unique()
        raise ValueError(f"final_result gagal dipetakan ke label: {list(bad)}")
    out["at_risk"] = out["at_risk"].astype(int)
    return out


def base_rate(labels: pd.DataFrame, label_col: str = "at_risk") -> float:
    """Proporsi at-risk (untuk dilaporkan per window pada tabel hasil)."""
    if len(labels) == 0:
        return float("nan")
    return float(labels[label_col].mean())
