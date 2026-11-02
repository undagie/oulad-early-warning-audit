"""Tabel inventaris fitur untuk paper — nama fitur ditarik LANGSUNG dari kode
(agar selalu sinkron), dipasangkan dengan deskripsi & sumber.

Output:
  - outputs/feature_inventory.csv
  - feature_inventory.md  (siap-tempel ke manuskrip, tabel per kelompok)

    python scripts/make_feature_inventory.py
"""

from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from nsoulad.features.assessment import ASSESSMENT_FEATURES
from nsoulad.features.course_structure import COURSE_STRUCTURE_FEATURES
from nsoulad.features.static import NUMERIC_COLS, ONEHOT_COLS, ORDINAL_ORDERS
from nsoulad.features.temporal import SEQ_FEATURES

# Deskripsi kuratif (dtype: num=numerik, bin=biner, cat=kategorik, ord=ordinal).
DESC: dict[str, tuple[str, str]] = {
    # Temporal (per minggu; K langkah untuk Week K)
    "clicks": ("num", "Total klik VLE pada minggu"),
    "active_days": ("num", "Jumlah hari aktif berbeda pada minggu"),
    "mean_clicks_active": ("num", "Rata-rata klik per hari aktif"),
    "n_activity_types": ("num", "Jumlah tipe aktivitas VLE berbeda"),
    "clicks_resource": ("num", "Klik pada materi tipe 'resource'"),
    "clicks_forumng": ("num", "Klik pada forum diskusi"),
    "clicks_homepage": ("num", "Klik pada homepage course"),
    "clicks_quiz": ("num", "Klik pada aktivitas kuis"),
    "clicks_subpage": ("num", "Klik pada subpage materi"),
    "is_inactive": ("bin", "Indikator minggu tanpa klik (clicks=0)"),
    "delta_clicks": ("num", "Perubahan klik vs minggu sebelumnya"),
    "prop_cumulative": ("num", "Proporsi klik minggu ini thd kumulatif"),
    # Assessment
    "n_submitted": ("num", "Jumlah asesmen yang sudah disubmit"),
    "first_score": ("num", "Skor asesmen pertama (deadline terawal)"),
    "mean_score": ("num", "Rata-rata skor asesmen yang disubmit"),
    "n_late": ("num", "Jumlah submission terlambat"),
    "mean_lateness": ("num", "Rata-rata keterlambatan (hari)"),
    "completion_ratio": ("num", "Rasio asesmen disubmit thd yang jatuh tempo"),
    "total_weight_done": ("num", "Total bobot asesmen yang terselesaikan"),
    "has_assessment": ("bin", "Indikator sudah punya asesmen jatuh tempo"),
    # Course-structure
    "scheduled_resource_ratio": ("num", "Rasio scheduled resources yang diakses"),
    "accessed_before_any_deadline": ("bin", "Aktif dalam 7 hari sebelum suatu deadline"),
    "active_days_7d_pre_deadline": ("num", "Hari aktif dalam 7 hari pra-deadline"),
    # Static ordinal/numeric
    "highest_education": ("ord", "Jenjang pendidikan tertinggi"),
    "age_band": ("ord", "Kelompok usia"),
    "imd_band": ("ord", "Index of Multiple Deprivation (batas bawah)"),
    "num_of_prev_attempts": ("num", "Jumlah percobaan module sebelumnya"),
    "studied_credits": ("num", "Total kredit yang sedang ditempuh"),
    "gender": ("cat", "Jenis kelamin (one-hot)"),
    "region": ("cat", "Region geografis (one-hot)"),
    "disability": ("cat", "Status disabilitas (one-hot)"),
    "code_module": ("cat", "Kode module (one-hot)"),
    "code_presentation": ("cat", "Kode presentation/tahun (one-hot)"),
}

# Sumber tabel OULAD per kelompok.
GROUPS = [
    ("Temporal (sekuensial, GRU)", "studentVle × vle", "per (mahasiswa, minggu); K langkah", SEQ_FEATURES),
    ("Assessment (tabular)", "studentAssessment × assessments", "per mahasiswa; deadline ≤ cutoff", ASSESSMENT_FEATURES),
    ("Course-structure (tabular)", "studentVle × vle, assessments", "per mahasiswa; struktur & deadline", COURSE_STRUCTURE_FEATURES),
    ("Static / demografi (tabular)", "studentInfo (tanpa final_result)", "per mahasiswa; fit-on-train",
     list(ORDINAL_ORDERS) + ["imd_band"] + NUMERIC_COLS + ONEHOT_COLS),
]


def main() -> int:
    rows = []
    md = ["# Feature Inventory\n",
          "Ditarik otomatis dari kode (`scripts/make_feature_inventory.py`); semua fitur",
          "leak-free (dipotong ≤ cutoff window, LEAP) dan — untuk static — encoder fit-on-train.\n"]
    for group, source, gran, feats in GROUPS:
        md.append(f"\n## {group}\n")
        md.append(f"*Sumber:* `{source}` — *granularitas:* {gran}\n")
        md.append("| Fitur | Tipe | Deskripsi |")
        md.append("|---|:--:|---|")
        for f in feats:
            dtype, desc = DESC.get(f, ("?", "(deskripsi belum ditulis)"))
            md.append(f"| `{f}` | {dtype} | {desc} |")
            rows.append({"group": group, "feature": f, "dtype": dtype,
                         "description": desc, "source": source, "granularity": gran})

    # Ringkasan hitungan.
    n_seq = len(SEQ_FEATURES)
    n_ass = len(ASSESSMENT_FEATURES)
    n_cs = len(COURSE_STRUCTURE_FEATURES)
    n_stat_base = len(list(ORDINAL_ORDERS) + ["imd_band"] + NUMERIC_COLS)
    n_onehot_groups = len(ONEHOT_COLS)
    summary = (f"\n## Ringkasan\n\n"
               f"- Temporal sekuensial: **{n_seq} fitur/minggu** "
               f"(diflatten mean/sum/last/slope = {n_seq*4} untuk model tree/linear).\n"
               f"- Assessment: **{n_ass}**; Course-structure: **{n_cs}**.\n"
               f"- Static: **{n_stat_base} ordinal/numerik** + one-hot dari {n_onehot_groups} kolom "
               f"(gender, region, disability, code_module, code_presentation).\n"
               f"- Symbolic: **7 rule** (R1–R7) → P_symbolic + rule_count + high_severity_count "
               f"(lihat `formulas.md` §4).\n")
    md.append(summary)

    out = Path("outputs"); out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "feature_inventory.csv", index=False)
    Path("feature_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"-> outputs/feature_inventory.csv ({len(rows)} fitur terdaftar)")
    print("-> feature_inventory.md")
    # Cek deskripsi yang belum ditulis.
    missing = [r["feature"] for r in rows if r["description"].startswith("(")]
    if missing:
        print(f"PERINGATAN: deskripsi belum ada untuk: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
