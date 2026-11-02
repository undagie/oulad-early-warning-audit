"""Test windowing LEAP: pemetaan minggu, cutoff, cohort, dan cutter cutoff-first."""

import numpy as np
import pandas as pd
import pytest

from nsoulad.data.labels import base_rate, make_binary_label
from nsoulad.data.schema import UNIT_KEYS
from nsoulad.data.windowing import (
    build_window_view,
    cohort_at_window,
    day_to_week,
    truncate_by_cutoff,
    week_cutoff_day,
)


def test_day_to_week_scalar():
    assert day_to_week(0) == 0
    assert day_to_week(6) == 0
    assert day_to_week(7) == 1
    assert day_to_week(13) == 1
    assert day_to_week(14) == 2
    assert day_to_week(-5) == -1  # pra-mulai


def test_day_to_week_series():
    s = pd.Series([0, 6, 7, 14])
    out = day_to_week(s)
    assert list(out) == [0, 0, 1, 2]


def test_week_cutoff_day():
    assert week_cutoff_day(2) == 13
    assert week_cutoff_day(4) == 27
    assert week_cutoff_day(6) == 41


def test_truncate_keeps_past_excludes_future_and_nan():
    df = pd.DataFrame({"date": [-5, 0, 13, 14, np.nan]})
    out = truncate_by_cutoff(df, cutoff_day=13)
    assert sorted(out["date"].dropna().tolist()) == [-5.0, 0.0, 13.0]
    assert out["date"].isna().sum() == 0  # NaN tereksklusi (timing tak diketahui)


def test_cohort_window_membership(oulad):
    week2 = set(cohort_at_window(oulad.student_registration, 2)["id_student"])
    assert week2 == {1, 2, 3, 5, 6}  # s4 belum registrasi pada cutoff13

    week4 = set(cohort_at_window(oulad.student_registration, 4)["id_student"])
    assert 4 in week4  # s4 registrasi hari 20 <= cutoff27

    week6 = set(cohort_at_window(oulad.student_registration, 6)["id_student"])
    assert 3 not in week6  # s3 unregister hari 30 <= cutoff41


def test_build_window_view_no_leakage(oulad):
    view = build_window_view(oulad, k=2)
    assert view.cutoff_day == 13
    assert view.n_students() == 5

    # Jaminan LEAP: tidak ada event setelah cutoff.
    assert (view.student_vle["date"] <= 13).all()
    assert (view.student_assessment["date_submitted"] <= 13).all()
    assert (view.assessments_due["date"] <= 13).all()

    # Anti-leakage: final_result tidak boleh ada di fitur statis.
    assert "final_result" not in view.static.columns


def test_build_window_view_counts(oulad):
    view = build_window_view(oulad, k=2)
    assert len(view.student_vle) == 6
    assert set(view.assessments_due["id_assessment"]) == {1001, 2001}
    assert len(view.student_assessment) == 2


def test_window_base_rate(oulad):
    labels = make_binary_label(oulad.student_info)
    view = build_window_view(oulad, k=2)
    cohort_labels = view.cohort_keys.merge(labels, on=list(UNIT_KEYS), how="left")
    # cohort week2 = {s1=0,s2=1,s3=1,s5=1,s6=0} -> 3/5
    assert base_rate(cohort_labels) == pytest.approx(0.6)
