"""Pembagian data leak-free: student-level stratified dan presentation-based.

Aturan kunci (lihat desain '## Data Split'):
  - student-level: mahasiswa yang sama TIDAK boleh muncul di train dan test;
  - stratified berdasarkan label at-risk pada level mahasiswa;
  - presentation-based: train pada presentation lama, test pada presentation baru.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _student_level_label(labels: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Label per id_student (max = 'pernah at-risk') untuk stratifikasi konsisten."""
    return (
        labels.groupby("id_student")[label_col]
        .max()
        .reset_index()
        .rename(columns={label_col: "_student_label"})
    )


def student_level_split(
    labels: pd.DataFrame,
    label_col: str = "at_risk",
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> SplitResult:
    """Split stratified berdasarkan id_student.

    `labels` minimal berisi kolom ['id_student', label_col] (boleh berisi baris
    per unit analisis). Pembagian dilakukan pada himpunan id_student unik agar
    tidak ada kebocoran mahasiswa antar split; lalu seluruh baris milik tiap
    mahasiswa mengikuti split-nya.
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios harus berjumlah 1.0, dapat {sum(ratios)}")

    rng = np.random.default_rng(seed)
    student_labels = _student_level_label(labels, label_col)

    train_ids: list = []
    val_ids: list = []
    test_ids: list = []
    r_train, r_val, _ = ratios

    # Stratifikasi: bagi tiap kelas secara terpisah agar proporsi terjaga.
    for _, grp in student_labels.groupby("_student_label"):
        ids = grp["id_student"].to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * r_train))
        n_val = int(round(n * r_val))
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train : n_train + n_val])
        test_ids.extend(ids[n_train + n_val :])

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)

    def _subset(ids: set) -> pd.DataFrame:
        return labels[labels["id_student"].isin(ids)].reset_index(drop=True)

    result = SplitResult(_subset(train_set), _subset(val_set), _subset(test_set))

    # Sanity: tidak ada mahasiswa yang bocor antar split.
    assert train_set.isdisjoint(test_set), "Kebocoran id_student antara train dan test"
    assert train_set.isdisjoint(val_set), "Kebocoran id_student antara train dan val"
    assert val_set.isdisjoint(test_set), "Kebocoran id_student antara val dan test"
    return result


def presentation_split(
    labels: pd.DataFrame,
    train_presentations: tuple[str, ...] = ("2013B", "2013J"),
    test_presentations: tuple[str, ...] = ("2014B", "2014J"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split berbasis code_presentation untuk uji generalisasi temporal."""
    if "code_presentation" not in labels.columns:
        raise ValueError("labels memerlukan kolom 'code_presentation' untuk presentation_split.")
    train = labels[labels["code_presentation"].isin(train_presentations)].reset_index(drop=True)
    test = labels[labels["code_presentation"].isin(test_presentations)].reset_index(drop=True)
    return train, test
