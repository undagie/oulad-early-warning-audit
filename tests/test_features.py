"""Test feature engineering (Fase 1) pada fixture sintetis terkontrol."""

import numpy as np
import pytest

from nsoulad.data.windowing import build_window_view
from nsoulad.features import (
    SEQ_FEATURES,
    StaticEncoder,
    assessment_features,
    build_window_dataset,
    flatten_sequence,
    presentation_split_datasets,
)
from nsoulad.features.temporal import weekly_engagement


def _row_index(keys, id_student):
    return keys.index[keys["id_student"] == id_student][0]


def test_weekly_no_future_weeks(oulad):
    view = build_window_view(oulad, k=4)  # K=4 -> minggu 0..3
    weekly = weekly_engagement(view)
    assert weekly["week"].max() <= 3
    assert weekly["week"].min() >= 0


def test_sequence_shape_and_known_values(oulad):
    view = build_window_view(oulad, k=4)
    ds = build_window_dataset(view, oulad.student_info, fit_static=True)
    n = ds.n
    assert ds.X_seq.shape == (n, 4, len(SEQ_FEATURES))

    # Mahasiswa 1: klik per minggu = [8, 2, 9, 0] (lihat conftest).
    i = _row_index(ds.keys, 1)
    clicks_idx = SEQ_FEATURES.index("clicks")
    assert list(ds.X_seq[i, :, clicks_idx]) == [8.0, 2.0, 9.0, 0.0]

    # forumng hanya pada minggu 1 (site 101, date 10).
    fg_idx = SEQ_FEATURES.index("clicks_forumng")
    assert ds.X_seq[i, 1, fg_idx] == 2.0


def test_tabular_no_nan_and_aligned(oulad):
    view = build_window_view(oulad, k=4)
    ds = build_window_dataset(view, oulad.student_info, fit_static=True)
    assert len(ds.X_tab) == ds.n
    assert not ds.X_tab.isna().any().any()
    assert ds.y.shape == (ds.n,)


def test_assessment_features_values(oulad):
    view = build_window_view(oulad, k=4)
    feats = assessment_features(view).set_index("id_student")
    # s1 submit assessment 1001 skor 80; 1 dari 1 due di course-nya.
    assert feats.loc[1, "first_score"] == 80.0
    assert feats.loc[1, "completion_ratio"] == pytest.approx(1.0)
    assert feats.loc[1, "has_assessment"] == 1.0
    # s3 tidak submit apa pun.
    assert feats.loc[3, "has_assessment"] == 0.0


def test_static_encoder_fit_on_train_only(oulad):
    view = build_window_view(oulad, k=4)
    static = view.static
    enc = StaticEncoder().fit(static)
    out = enc.transform(static)
    # Tidak ada NaN dan jumlah kolom konsisten.
    assert not out.isna().any().any()
    assert list(out.columns) == enc.feature_names_


def test_presentation_split_datasets(oulad):
    view = build_window_view(oulad, k=4)
    parts = presentation_split_datasets(
        view, oulad.student_info,
        train_presentations=("2013B", "2013J"), test_presentations=("2014B", "2014J"),
    )
    # Test set = mahasiswa presentation 2014J (s5, s6).
    assert set(parts["test"].keys["id_student"]) == {5, 6}
    # Train/val hanya dari presentation 2013.
    trainval = set(parts["train"].keys["id_student"]) | set(parts["val"].keys["id_student"])
    assert trainval <= {1, 2, 3, 4}


def test_flatten_sequence_shape(oulad):
    view = build_window_view(oulad, k=4)
    ds = build_window_dataset(view, oulad.student_info, fit_static=True)
    flat = flatten_sequence(ds.X_seq, ds.seq_features)
    assert len(flat) == ds.n
    assert flat.shape[1] == len(SEQ_FEATURES) * 4  # mean/sum/last/slope
