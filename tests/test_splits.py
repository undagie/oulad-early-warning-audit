"""Test pembagian data leak-free."""

import pandas as pd

from nsoulad.data.labels import make_binary_label
from nsoulad.data.splits import presentation_split, student_level_split


def test_student_split_no_leakage(oulad):
    labels = make_binary_label(oulad.student_info)
    res = student_level_split(labels, seed=0)
    train_ids = set(res.train["id_student"])
    val_ids = set(res.val["id_student"])
    test_ids = set(res.test["id_student"])
    assert train_ids.isdisjoint(test_ids)
    assert train_ids.isdisjoint(val_ids)
    assert val_ids.isdisjoint(test_ids)
    # Semua mahasiswa terdistribusi (tidak ada yang hilang).
    assert train_ids | val_ids | test_ids == set(labels["id_student"])


def test_student_split_deterministic(oulad):
    labels = make_binary_label(oulad.student_info)
    a = student_level_split(labels, seed=123)
    b = student_level_split(labels, seed=123)
    assert set(a.train["id_student"]) == set(b.train["id_student"])


def test_presentation_split(oulad):
    labels = make_binary_label(oulad.student_info)
    train, test = presentation_split(labels)
    assert set(train["code_presentation"]) <= {"2013B", "2013J"}
    assert set(test["code_presentation"]) <= {"2014B", "2014J"}
    assert set(test["id_student"]) == {5, 6}
