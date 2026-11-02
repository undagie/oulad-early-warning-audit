"""Rakit WindowView -> WindowDataset (tensor sekuens + matriks tabular + label).

Komponen:
  - X_seq  : tensor temporal [n, K, F] untuk GRU.
  - X_tab  : matriks tabular (static + assessment + course-structure) untuk fusion & tree.
  - y      : label at-risk.
  - keys   : UNIT_KEYS selaras urutan baris (untuk join hasil/symbolic).

StaticEncoder di-fit pada train (fit_static=True) lalu dipakai-ulang di val/test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nsoulad.data.labels import make_binary_label
from nsoulad.data.schema import UNIT_KEYS
from nsoulad.data.splits import student_level_split
from nsoulad.data.windowing import WindowView
from nsoulad.features.assessment import ASSESSMENT_FEATURES, assessment_features
from nsoulad.features.course_structure import (
    COURSE_STRUCTURE_FEATURES,
    course_structure_features,
)
from nsoulad.features.static import StaticEncoder
from nsoulad.features.temporal import to_sequence_tensor, weekly_engagement

_KEYS = list(UNIT_KEYS)


@dataclass
class WindowDataset:
    k: int
    keys: pd.DataFrame          # UNIT_KEYS, urutan baris = urutan X_seq/X_tab/y
    X_seq: np.ndarray           # [n, K, F]
    seq_features: list[str]
    X_tab: pd.DataFrame         # fitur tabular non-sekuensial
    y: np.ndarray               # [n]
    encoder: StaticEncoder

    @property
    def n(self) -> int:
        return len(self.keys)

    def subset(self, mask: np.ndarray) -> "WindowDataset":
        """Iris dataset berdasarkan mask boolean baris (untuk train/val/test)."""
        return WindowDataset(
            k=self.k,
            keys=self.keys.loc[mask].reset_index(drop=True),
            X_seq=self.X_seq[mask],
            seq_features=self.seq_features,
            X_tab=self.X_tab.loc[mask].reset_index(drop=True),
            y=self.y[mask],
            encoder=self.encoder,
        )


def build_window_dataset(
    view: WindowView,
    student_info: pd.DataFrame,
    encoder: StaticEncoder | None = None,
    fit_static: bool = False,
    days_per_week: int = 7,
) -> WindowDataset:
    """Bangun WindowDataset untuk satu window.

    student_info dipakai untuk label (make_binary_label). encoder/fit_static
    mengatur fit-on-train: pada train berikan encoder=None, fit_static=True; pada
    val/test berikan encoder hasil train, fit_static=False.
    """
    # Urutan baris kanonik = cohort_keys.
    keys = view.cohort_keys[_KEYS].drop_duplicates().reset_index(drop=True)

    # --- Sekuens temporal ---
    weekly = weekly_engagement(view, days_per_week=days_per_week)
    X_seq, seq_features, order = to_sequence_tensor(weekly, keys, view.k)
    keys = order.reset_index(drop=True)  # pastikan urutan konsisten dgn tensor

    # --- Tabular: static (encoded) + assessment + course-structure ---
    # keys sebagai frame KIRI menjamin urutan baris konsisten dgn X_seq/y.
    static_raw = keys.merge(view.static, on=_KEYS, how="left")
    if fit_static or encoder is None:
        encoder = StaticEncoder().fit(static_raw)
    X_static = encoder.transform(static_raw)

    assess = keys.merge(assessment_features(view), on=_KEYS, how="left")
    course = keys.merge(course_structure_features(view, days_per_week=days_per_week), on=_KEYS, how="left")

    X_tab = pd.concat(
        [
            X_static.reset_index(drop=True),
            assess[ASSESSMENT_FEATURES].reset_index(drop=True),
            course[COURSE_STRUCTURE_FEATURES].reset_index(drop=True),
        ],
        axis=1,
    )

    # --- Label ---
    labels = make_binary_label(student_info)
    y_df = keys.merge(labels[_KEYS + ["at_risk"]], on=_KEYS, how="left")
    if y_df["at_risk"].isna().any():
        missing = int(y_df["at_risk"].isna().sum())
        raise ValueError(f"{missing} mahasiswa cohort tanpa label — periksa studentInfo.")
    y = y_df["at_risk"].to_numpy(dtype=int)

    return WindowDataset(
        k=view.k,
        keys=keys,
        X_seq=X_seq,
        seq_features=seq_features,
        X_tab=X_tab,
        y=y,
        encoder=encoder,
    )


def split_window_datasets(
    view: WindowView,
    student_info: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    days_per_week: int = 7,
) -> dict[str, WindowDataset]:
    """Bangun dataset train/val/test untuk satu window, encoder fit-on-train (leak-free).

    Split berbasis id_student (tidak ada mahasiswa bocor antar split). Encoder
    statis hanya di-fit pada baris train, lalu dipakai-ulang untuk val/test.
    """
    labels = make_binary_label(student_info)
    cohort_labels = view.cohort_keys.merge(labels[_KEYS + ["at_risk"]], on=_KEYS, how="left")
    split = student_level_split(cohort_labels[_KEYS + ["at_risk"]], ratios=ratios, seed=seed)
    train_ids = set(split.train["id_student"])
    val_ids = set(split.val["id_student"])
    test_ids = set(split.test["id_student"])

    # Encoder fit pada static train saja.
    static_train = view.static[view.static["id_student"].isin(train_ids)]
    encoder = StaticEncoder().fit(static_train)

    ds = build_window_dataset(
        view, student_info, encoder=encoder, fit_static=False, days_per_week=days_per_week
    )

    ids = ds.keys["id_student"].to_numpy()
    masks = {
        "train": np.isin(ids, list(train_ids)),
        "val": np.isin(ids, list(val_ids)),
        "test": np.isin(ids, list(test_ids)),
    }
    return {name: ds.subset(mask) for name, mask in masks.items()}


def presentation_split_datasets(
    view: WindowView,
    student_info: pd.DataFrame,
    train_presentations: tuple[str, ...] = ("2013B", "2013J"),
    test_presentations: tuple[str, ...] = ("2014B", "2014J"),
    val_frac: float = 0.15,
    seed: int = 42,
    days_per_week: int = 7,
    exclude_returning: bool = False,
) -> dict[str, WindowDataset]:
    """Dataset train/val/test berbasis presentation (uji generalisasi temporal).

    Train+val diambil dari train_presentations (val di-carve acak per mahasiswa),
    test dari test_presentations. Encoder fit pada train saja (leak-free).

    exclude_returning=True: buang dari test setiap mahasiswa yang punya enrolment apa pun
    di train_presentations (menurut studentInfo), sehingga cohort test identity-disjoint.
    """
    keys = view.cohort_keys[_KEYS].drop_duplicates().reset_index(drop=True)
    pres = keys["code_presentation"]
    test_mask = pres.isin(test_presentations).to_numpy()
    if exclude_returning:
        seen = student_info.loc[student_info["code_presentation"].isin(train_presentations), "id_student"]
        test_mask &= ~keys["id_student"].isin(set(seen)).to_numpy()
    trainval_mask = pres.isin(train_presentations).to_numpy()

    # Carve val dari train_presentations secara student-level.
    tv_keys = keys[trainval_mask]
    labels = make_binary_label(student_info)
    tv_labels = tv_keys.merge(labels[_KEYS + ["at_risk"]], on=_KEYS, how="left")
    inner = student_level_split(tv_labels[_KEYS + ["at_risk"]],
                                ratios=(1 - val_frac, val_frac, 0.0), seed=seed)
    train_ids = set(inner.train["id_student"])

    static_train = view.static[view.static["id_student"].isin(train_ids)
                               & view.static["code_presentation"].isin(train_presentations)]
    encoder = StaticEncoder().fit(static_train)
    ds = build_window_dataset(view, student_info, encoder=encoder, fit_static=False,
                              days_per_week=days_per_week)

    ids = ds.keys["id_student"].to_numpy()
    if exclude_returning:  # ds.keys bisa berurutan beda dari keys; hitung ulang mask test
        seen = set(student_info.loc[student_info["code_presentation"].isin(train_presentations), "id_student"])
        test_mask = ds.keys["code_presentation"].isin(test_presentations).to_numpy() & ~np.isin(ids, list(seen))
    else:
        test_mask = ds.keys["code_presentation"].isin(test_presentations).to_numpy()
    trainval_mask = ds.keys["code_presentation"].isin(train_presentations).to_numpy()
    val_mask = trainval_mask & np.isin(ids, list(inner.val["id_student"]))
    train_mask = trainval_mask & ~val_mask
    return {
        "train": ds.subset(train_mask),
        "val": ds.subset(val_mask),
        "test": ds.subset(test_mask),
    }


def module_holdout_datasets(
    view: WindowView,
    student_info: pd.DataFrame,
    module: str,
    val_frac: float = 0.15,
    seed: int = 42,
    days_per_week: int = 7,
) -> dict[str, WindowDataset]:
    """Leave-one-module-out: test = semua enrolment modul `module`, train/val = modul lain.

    Mahasiswa yang punya enrolment di modul test dikeluarkan dari train/val (identity-
    disjoint). Encoder statis di-fit pada train saja, sehingga one-hot modul test bernilai
    nol (kondisi modul baru). Val di-carve per mahasiswa dari modul non-test.
    """
    keys = view.cohort_keys[_KEYS].drop_duplicates().reset_index(drop=True)
    test_students = set(keys.loc[keys["code_module"] == module, "id_student"])
    dev_keys = keys[(keys["code_module"] != module) & ~keys["id_student"].isin(test_students)]
    labels = make_binary_label(student_info)
    dev_labels = dev_keys.merge(labels[_KEYS + ["at_risk"]], on=_KEYS, how="left")
    inner = student_level_split(dev_labels[_KEYS + ["at_risk"]],
                                ratios=(1 - val_frac, val_frac, 0.0), seed=seed)
    train_ids, val_ids = set(inner.train["id_student"]), set(inner.val["id_student"])

    static_train = view.static[view.static["id_student"].isin(train_ids)
                               & (view.static["code_module"] != module)]
    encoder = StaticEncoder().fit(static_train)
    ds = build_window_dataset(view, student_info, encoder=encoder, fit_static=False,
                              days_per_week=days_per_week)
    ids = ds.keys["id_student"].to_numpy()
    mod = ds.keys["code_module"].to_numpy()
    dev = (mod != module) & ~np.isin(ids, list(test_students))
    return {
        "train": ds.subset(dev & np.isin(ids, list(train_ids))),
        "val": ds.subset(dev & np.isin(ids, list(val_ids))),
        "test": ds.subset(mod == module),
    }


def flatten_sequence(X_seq: np.ndarray, seq_features: list[str]) -> pd.DataFrame:
    """Ringkas tensor [n,K,F] -> fitur datar (mean/sum/last/slope) untuk model tree/linear."""
    n, K, F = X_seq.shape
    mean = X_seq.mean(axis=1)
    total = X_seq.sum(axis=1)
    last = X_seq[:, -1, :]
    # slope sederhana: selisih minggu terakhir vs pertama / (K-1)
    slope = (X_seq[:, -1, :] - X_seq[:, 0, :]) / max(K - 1, 1)

    cols = {}
    for j, name in enumerate(seq_features):
        cols[f"{name}_mean"] = mean[:, j]
        cols[f"{name}_sum"] = total[:, j]
        cols[f"{name}_last"] = last[:, j]
        cols[f"{name}_slope"] = slope[:, j]
    return pd.DataFrame(cols)
