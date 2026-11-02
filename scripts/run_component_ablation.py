"""Ablation course-structure rules (Tabel 4 / H4).

Bandingkan NS-GRU dengan SELURUH rule vs TANPA course-structure rules (R3, R7).
Fokus: AUC, recall, explanation coverage, dan contradiction-rate (H4 memprediksi
course-structure rules menurunkan contradiction-rate).

    python scripts/run_component_ablation.py --windows 4 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from nsoulad.config import load_config
from nsoulad.data.loader import load_oulad
from nsoulad.pipeline import run_window
from nsoulad.symbolic.rules import COURSE_STRUCTURE_RULES, RULE_IDS


def _ns_row(res: dict, label: str, k: int) -> dict:
    m = res["metrics"].set_index("model")
    ex = res.get("explanation", {})
    return {
        "window": k, "variant": label,
        "ns_auc": round(float(m.loc["NS-GRU", "auc"]), 4),
        "ns_recall": round(float(m.loc["NS-GRU", "recall"]), 4),
        "coverage": round(float(ex.get("coverage", float("nan"))), 4),
        "contradiction_rate": round(float(ex.get("contradiction_rate", float("nan"))), 4),
    }


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(cfg.data_dir))
    ap.add_argument("--windows", type=int, nargs="+", default=list(cfg.windows))
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = load_oulad(args.data_dir)
    no_cs = [r for r in RULE_IDS if r not in COURSE_STRUCTURE_RULES]

    rows = []
    for k in args.windows:
        full = run_window(tables, k, args.seed, cfg, include_gru=True, active_rules=None)
        ablat = run_window(tables, k, args.seed, cfg, include_gru=True, active_rules=no_cs)
        rows.append(_ns_row(full, "full (R1-R7)", k))
        rows.append(_ns_row(ablat, "no course-structure (-R3,R7)", k))

    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation_course_structure.csv", index=False)
    print("=== Ablation course-structure rules (NS-GRU) ===")
    print(df.to_string(index=False))
    print(f"\nDisimpan ke {out / 'ablation_course_structure.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
