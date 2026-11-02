"""Bangun WindowView LEAP-aligned untuk tiap prediction window dan cetak ringkasannya.

Jalankan:
    python scripts/build_windows.py [--data-dir data/raw]

Memvalidasi pipeline secara end-to-end pada data nyata: ukuran cohort, base rate
at-risk per window, jumlah event VLE setelah cutoff, dan jumlah assessment due.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

import pandas as pd

from nsoulad.config import load_config
from nsoulad.data.labels import base_rate, make_binary_label
from nsoulad.data.loader import load_oulad
from nsoulad.data.windowing import build_window_view
from nsoulad.data.schema import UNIT_KEYS


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Bangun window views OULAD")
    parser.add_argument("--data-dir", default=str(cfg.data_dir))
    args = parser.parse_args()

    tables = load_oulad(args.data_dir)
    labels = make_binary_label(
        tables.student_info, at_risk=cfg.label.at_risk, not_at_risk=cfg.label.not_at_risk
    )

    rows = []
    for k in cfg.windows:
        view = build_window_view(tables, k, days_per_week=cfg.days_per_week)
        cohort_labels = view.cohort_keys.merge(labels, on=list(UNIT_KEYS), how="left")
        rows.append(
            {
                "window": k,
                "cutoff_day": view.cutoff_day,
                "n_students": view.n_students(),
                "base_rate_at_risk": round(base_rate(cohort_labels), 4),
                "vle_events": len(view.student_vle),
                "assessments_due": len(view.assessments_due),
                "submissions": len(view.student_assessment),
            }
        )

    print("Ringkasan window (LEAP-aligned):")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
