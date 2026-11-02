"""Revisi SOIC 4706: akuntansi cohort per window (aturan eksplisit untuk withdrawal pra-cutoff).

Untuk tiap window: enrolment yang terdaftar sampai cutoff, yang dikeluarkan karena sudah
unregister sebelum/pada cutoff (outcome Withdrawn sudah teramati), ukuran cohort yang
dinilai, base rate, dan ukuran cohort tetap (terdaftar s.d. Week 2, aktif s.d. Week 8).

    python scripts/rev_cohort_accounting.py
"""

from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from nsoulad.config import load_config
from nsoulad.data.labels import make_binary_label
from nsoulad.data.loader import load_oulad
from nsoulad.data.windowing import cohort_at_window, week_cutoff_day

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["code_module", "code_presentation", "id_student"]


def main() -> int:
    cfg = load_config()
    t = load_oulad(str(ROOT / cfg.data_dir))
    reg = t.student_registration.merge(make_binary_label(t.student_info)[KEYS + ["final_result", "at_risk"]],
                                       on=KEYS, how="left")
    n_withdrawn = int((reg.final_result == "Withdrawn").sum())
    rows = []
    for k in cfg.windows:
        c = week_cutoff_day(k, cfg.days_per_week)
        registered = reg[reg.date_registration.notna() & (reg.date_registration <= c)]
        early = registered[registered.date_unregistration.notna() & (registered.date_unregistration <= c)]
        cohort = cohort_at_window(t.student_registration, k, cfg.days_per_week).merge(reg[KEYS + ["at_risk"]], on=KEYS)
        fixed = cohort_at_window(t.student_registration, k, cfg.days_per_week, fixed_weeks=(2, 8)).merge(
            reg[KEYS + ["at_risk"]], on=KEYS)
        rows.append({
            "window": k, "cutoff_day": c, "registered_by_cutoff": len(registered),
            "excluded_withdrawn_before_cutoff": len(early),
            # share of all Withdrawn outcomes that the rule removes (a few unregistered enrolments
            # carry another final result, so they are not counted here)
            "excluded_share_of_all_withdrawn": int((early.final_result == "Withdrawn").sum()) / n_withdrawn,
            "excluded_final_result_withdrawn": int((early.final_result == "Withdrawn").sum()),
            "scored_cohort": len(cohort), "scored_base_rate": float(cohort.at_risk.mean()),
            "fixed_cohort": len(fixed), "fixed_base_rate": float(fixed.at_risk.mean()),
        })
    df = pd.DataFrame(rows)
    out = ROOT / "outputs" / "revision" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cohort_accounting.csv", index=False)
    print(f"All Withdrawn enrolments: {n_withdrawn}")
    print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
