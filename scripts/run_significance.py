"""Uji signifikansi pairwise pada test set nyata.

Untuk tiap kondisi (window, fraksi data), latih model lalu uji:
  - paired bootstrap selisih AUC (NS-GRU vs pembanding);
  - McNemar (klasifikasi, threshold dituning di val).

Klaim yang divalidasi:
  - low-data (Week 2, 1%): NS-GRU > GRU & > XGBoost signifikan?  (data-efficiency)
  - full data: NS-GRU vs GRU TIDAK signifikan (non-inferiority H1);
              NS-GRU > Static-only signifikan (nilai fitur temporal).

    python scripts/run_significance.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nsoulad.config import load_config  # noqa: E402
from nsoulad.data.loader import load_oulad  # noqa: E402
from nsoulad.data.windowing import build_window_view  # noqa: E402
from nsoulad.evaluation.stats import holm_bonferroni, mcnemar_test, paired_bootstrap_auc_diff  # noqa: E402
from nsoulad.features.build import split_window_datasets  # noqa: E402
from nsoulad.imbalance import tune_threshold  # noqa: E402
from nsoulad.models import (  # noqa: E402
    GRUModel, LogisticFusion, StaticOnly, XGBoostFull, align_symbolic,
)
from nsoulad.symbolic import RuleThresholds, SymbolicEngine  # noqa: E402

CONDITIONS = [("Week2 low-data (1%)", 2, 0.01), ("Week2 full", 2, 1.0), ("Week8 full", 8, 1.0)]
COMPARISONS = [("NS-GRU", "GRU"), ("NS-GRU", "XGBoost"), ("NS-GRU", "Static-only")]


def _mask(y, frac, seed):
    if frac >= 1.0:
        return np.ones(len(y), bool)
    rng = np.random.default_rng(seed)
    m = np.zeros(len(y), bool)
    for c in (0, 1):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        m[idx[: max(2, int(round(len(idx) * frac)))]] = True
    return m


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(cfg.data_dir))
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    tables = load_oulad(args.data_dir)
    rows = []

    for label, k, frac in CONDITIONS:
        view = build_window_view(tables, k, days_per_week=cfg.days_per_week)
        parts = split_window_datasets(view, tables.student_info, ratios=cfg.split.ratios,
                                      seed=args.seed, days_per_week=cfg.days_per_week)
        train, val, test = parts["train"], parts["val"], parts["test"]
        tr = train.subset(_mask(train.y, frac, args.seed))

        th = RuleThresholds.fit(view, set(tr.keys["id_student"]), days_per_week=cfg.days_per_week)
        sf = SymbolicEngine(th, days_per_week=cfg.days_per_week).apply(view).to_frame()
        sym_val, sym_test = align_symbolic(sf, val.keys), align_symbolic(sf, test.keys)

        gru = GRUModel(seed=args.seed).fit(tr, val)
        pv_gru, pt_gru = gru.predict_proba(val), gru.predict_proba(test)
        fusion = LogisticFusion(seed=args.seed).fit(pv_gru, sym_val, val.y)
        xgb = XGBoostFull(seed=args.seed).fit(tr)
        sto = StaticOnly(seed=args.seed).fit(tr)

        probs = {
            "NS-GRU": (fusion.predict_proba(pv_gru, sym_val), fusion.predict_proba(pt_gru, sym_test)),
            "GRU": (pv_gru, pt_gru),
            "XGBoost": (xgb.predict_proba(val), xgb.predict_proba(test)),
            "Static-only": (sto.predict_proba(val), sto.predict_proba(test)),
        }
        preds = {}
        for name, (pv, pt) in probs.items():
            t = tune_threshold(val.y, pv)
            preds[name] = (pt >= t).astype(int)

        for a, b in COMPARISONS:
            bs = paired_bootstrap_auc_diff(test.y, probs[a][1], probs[b][1], n_boot=2000, seed=0)
            mc = mcnemar_test(test.y, preds[a], preds[b])
            rows.append({
                "condition": label, "n_test": test.n, "comparison": f"{a} vs {b}",
                "auc_diff": round(bs["auc_diff"], 4), "bootstrap_p": round(bs["p_value"], 4),
                "mcnemar_p": round(mc["p_value"], 4),
                "sig_0.05": "yes" if bs["p_value"] < 0.05 else "no",
            })
        print(f"[done] {label}")

    df = pd.DataFrame(rows)
    # Koreksi multiple-comparison (Holm-Bonferroni) atas seluruh keluarga uji bootstrap.
    holm = holm_bonferroni(df["bootstrap_p"].tolist(), alpha=0.05)
    df["p_holm"] = [round(p, 4) for p in holm["p_adjusted"]]
    df["sig_holm"] = ["yes" if r else "no" for r in holm["reject"]]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "significance.csv", index=False)
    print("\n=== Uji signifikansi pairwise (paired bootstrap AUC + McNemar) ===")
    print(df.to_string(index=False))
    print(f"\nDisimpan ke {out/'significance.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
