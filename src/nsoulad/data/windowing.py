"""Windowing LEAP-aligned: pemetaan minggu, cutoff, cohort, dan cutter cutoff-first.

Prinsip LEAP (arXiv 2605.25794): TRUNCATE setiap sumber data sampai cutoff
SEBELUM join/agregasi. Modul ini hanya menghasilkan *view* terpotong yang aman;
feature engineering dilakukan di tahap berikutnya di atas view ini.

Pemetaan minggu (lihat desain-eksperimen):
    week = floor(date / days_per_week)
    Week k mencakup hari 0 .. days_per_week*k - 1, jadi cutoff_day = days_per_week*k - 1.
    Aktivitas pra-mulai (date < 0) tetap dipertahankan (diagregasi ke Week 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nsoulad.data.loader import OuladTables
from nsoulad.data.schema import UNIT_KEYS


def day_to_week(date: pd.Series | np.ndarray | float, days_per_week: int = 7):
    """week = floor(date / days_per_week). Mendukung skalar maupun Series."""
    if isinstance(date, pd.Series):
        return np.floor(date / days_per_week).astype("Int64")
    return int(np.floor(date / days_per_week))


def week_cutoff_day(k: int, days_per_week: int = 7) -> int:
    """Hari cutoff terakhir yang termasuk Week k (inklusif)."""
    return days_per_week * k - 1


def truncate_by_cutoff(
    df: pd.DataFrame, cutoff_day: int, date_col: str = "date"
) -> pd.DataFrame:
    """Pertahankan baris dengan date_col <= cutoff_day (cutoff-first).

    Baris dengan date_col NaN otomatis tereksklusi (perbandingan NaN -> False),
    yang benar secara LEAP: timing tak diketahui dianggap belum tersedia.
    Baris dengan date negatif (pra-mulai) tetap dipertahankan.
    """
    if date_col not in df.columns:
        raise KeyError(f"Kolom tanggal '{date_col}' tidak ada pada DataFrame.")
    keep = df[date_col] <= cutoff_day
    return df.loc[keep.fillna(False)].copy()


def cohort_at_window(
    student_registration: pd.DataFrame, k: int, days_per_week: int = 7,
    fixed_weeks: tuple[int, int] | None = None,
    include_withdrawn: bool = False,
) -> pd.DataFrame:
    """Cohort window ke-k: mahasiswa yang masih *registered* pada cutoff.

    Kriteria (lihat desain, '## Target Label' -> definisi cohort per window):
      - sudah registrasi pada/<= cutoff: date_registration <= cutoff_day; dan
      - belum unregister pada cutoff: date_unregistration NaN atau > cutoff_day.

    Enrolment yang sudah unregister sebelum cutoff dikeluarkan karena hasilnya
    (Withdrawn) sudah teramati saat prediksi dibuat, jadi tidak perlu diprediksi.

    fixed_weeks=(a, b): cohort tetap untuk analisis sensitivitas komposisi cohort —
    registrasi paling lambat akhir Week a dan masih registered pada akhir Week b,
    sehingga enrolment yang sama dinilai di setiap window antara a dan b.

    include_withdrawn=True: cohort didefinisikan saat enrolment — semua yang terdaftar sampai
    cutoff, TERMASUK yang sudah unregister (seperti benchmark LEAP). Hanya untuk analisis
    sensitivitas; enrolment itu sudah ber-outcome Withdrawn sehingga mudah ditebak.

    Returns subset baris registrasi (UNIT_KEYS + tanggal) untuk cohort tersebut.
    """
    if fixed_weeks is None:
        reg_cut = unreg_cut = week_cutoff_day(k, days_per_week)
    else:
        reg_cut = week_cutoff_day(fixed_weeks[0], days_per_week)
        unreg_cut = week_cutoff_day(fixed_weeks[1], days_per_week)
    reg = student_registration

    registered_by = reg["date_registration"].notna() & (reg["date_registration"] <= reg_cut)
    not_unregistered = reg["date_unregistration"].isna() | (reg["date_unregistration"] > unreg_cut)
    if include_withdrawn:
        not_unregistered = pd.Series(True, index=reg.index)

    cohort = reg.loc[registered_by & not_unregistered].copy()
    return cohort.reset_index(drop=True)


@dataclass
class WindowView:
    """View data yang sudah dipotong cutoff-first untuk satu window.

    Semua tabel di sini dijamin bebas-bocor: tidak ada record dengan timing
    setelah cutoff_day, dan hanya memuat cohort window tersebut.
    """

    k: int
    cutoff_day: int
    cohort_keys: pd.DataFrame          # UNIT_KEYS untuk cohort window ini
    static: pd.DataFrame               # studentInfo cohort, TANPA final_result
    registration: pd.DataFrame         # registrasi cohort
    student_vle: pd.DataFrame          # aktivitas VLE <= cutoff, cohort
    assessments_due: pd.DataFrame      # assessment dengan deadline <= cutoff
    student_assessment: pd.DataFrame   # submission <= cutoff utk assessment due
    vle: pd.DataFrame                  # metadata VLE untuk course cohort
    courses: pd.DataFrame              # metadata course untuk cohort

    def n_students(self) -> int:
        return len(self.cohort_keys)


def _filter_to_keys(df: pd.DataFrame, keys: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    """Inner-join filter df agar hanya memuat baris yang cocok dengan keys."""
    return df.merge(keys[on].drop_duplicates(), on=on, how="inner")


def build_window_view(
    tables: OuladTables, k: int, days_per_week: int = 7,
    feature_cutoff_day: int | None = None,
    score_lag_days: int = 0,
    fixed_cohort_weeks: tuple[int, int] | None = None,
    include_withdrawn: bool = False,
) -> WindowView:
    """Bangun WindowView LEAP-aligned untuk Week k (cutoff-first, lalu filter cohort).

    Urutan operasi penting (LEAP): tentukan cohort & cutoff -> TRUNCATE tiap
    sumber -> baru filter ke cohort/course. `final_result` sengaja dibuang dari
    fitur statis (anti-leakage); label diambil terpisah via labels.make_binary_label.

    feature_cutoff_day: jika diberikan, fitur (VLE/assessment) dipotong pada hari ini
    alih-alih week_cutoff_day(k). Dipakai untuk ablation leakage: nilai besar -> versi
    "leaky" (fitur melihat data masa depan) sementara cohort tetap di window k.

    score_lag_days: sensitivitas visibilitas skor. OULAD mencatat kapan submission dibuat,
    tetapi tidak kapan skornya dirilis. Dengan lag L > 0, skor dianggap terlihat mulai
    max(hari submit, deadline) + L; skor yang belum terlihat di-set NaN (submission,
    waktunya, dan keterlambatannya tetap teramati). L = 0 = analisis utama (skor
    dianggap terlihat saat submit).

    fixed_cohort_weeks: lihat cohort_at_window (sensitivitas komposisi cohort).
    """
    cutoff = week_cutoff_day(k, days_per_week)
    fcutoff = cutoff if feature_cutoff_day is None else feature_cutoff_day

    # 1) Cohort window ke-k (kunci unit analisis).
    cohort_reg = cohort_at_window(tables.student_registration, k, days_per_week,
                                  fixed_weeks=fixed_cohort_weeks, include_withdrawn=include_withdrawn)
    cohort_keys = cohort_reg[list(UNIT_KEYS)].drop_duplicates().reset_index(drop=True)

    # 2) Static features: studentInfo cohort TANPA final_result.
    static = _filter_to_keys(tables.student_info, cohort_keys, list(UNIT_KEYS))
    static = static.drop(columns=["final_result"], errors="ignore")

    # 3) studentVle: TRUNCATE date <= fcutoff, lalu filter cohort.
    svle = truncate_by_cutoff(tables.student_vle, fcutoff, date_col="date")
    svle = _filter_to_keys(svle, cohort_keys, list(UNIT_KEYS))

    # 4) assessments: hanya yang deadline-nya sudah jatuh (date <= fcutoff),
    #    difilter ke course milik cohort.
    course_keys = cohort_keys[["code_module", "code_presentation"]].drop_duplicates()
    assess_due = truncate_by_cutoff(tables.assessments, fcutoff, date_col="date")
    assess_due = assess_due.merge(course_keys, on=["code_module", "code_presentation"], how="inner")

    # 5) studentAssessment: TRUNCATE submission <= fcutoff, lalu batasi ke
    #    assessment yang due-by-cutoff dan ke mahasiswa cohort.
    sa = truncate_by_cutoff(tables.student_assessment, fcutoff, date_col="date_submitted")
    sa = sa.merge(assess_due[["id_assessment"]].drop_duplicates(), on="id_assessment", how="inner")
    sa = sa.merge(cohort_keys[["id_student"]].drop_duplicates(), on="id_student", how="inner")
    if score_lag_days:
        deadline = assess_due[["id_assessment", "date"]].drop_duplicates("id_assessment")
        deadline = deadline.rename(columns={"date": "_deadline"})
        sa = sa.merge(deadline, on="id_assessment", how="left")
        visible_day = np.maximum(sa["date_submitted"].astype(float),
                                 sa["_deadline"].astype(float).fillna(-np.inf)) + score_lag_days
        sa["score"] = sa["score"].where(visible_day <= fcutoff)
        sa = sa.drop(columns="_deadline")

    # 6) vle & courses: metadata course cohort (schedule, tidak ada timing event).
    vle = tables.vle.merge(course_keys, on=["code_module", "code_presentation"], how="inner")
    courses = tables.courses.merge(course_keys, on=["code_module", "code_presentation"], how="inner")

    return WindowView(
        k=k,
        cutoff_day=cutoff,
        cohort_keys=cohort_keys,
        static=static,
        registration=cohort_reg,
        student_vle=svle,
        assessments_due=assess_due,
        student_assessment=sa,
        vle=vle,
        courses=courses,
    )
