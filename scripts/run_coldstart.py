"""Cold-start (leave-one-module-out) — the ONE accuracy-facing regime where the symbolic
prior is expected to help (a brand-new module with no history).

For each held-out module m: train on all OTHER modules, test on m. The static encoder is fit
on non-m data, so m's module one-hot is all-zero (module identity is uninformative — exactly
the cold-start condition). We compare NS-GRU vs GRU vs XGBoost vs Rules-only by test ROC-AUC.

Pre-registered expectation: NS-GRU > GRU/XGBoost on held-out modules, largest at Week 2.
Output: outputs/coldstart.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from nsoulad.config import load_config
from nsoulad.data.loader import load_oulad
from nsoulad.data.windowing import build_window_view
from nsoulad.features.build import build_window_dataset
from nsoulad.features.static import StaticEncoder
from nsoulad.models import GRUModel, LogisticFusion, XGBoostFull, align_symbolic
from nsoulad.symbolic import RuleThresholds, SymbolicEngine

WINDOWS = [2, 8]
SEED = 42


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def main():
    cfg = load_config()
    tables = load_oulad(str(cfg.data_dir))
    rows = []

    for k in WINDOWS:
        view = build_window_view(tables, k, days_per_week=cfg.days_per_week)
        modules = sorted(view.cohort_keys["code_module"].unique())
        for m in modules:
            # Encoder fit on NON-held-out modules (module m one-hot -> all zero => cold-start).
            static_dev = view.static[view.static["code_module"] != m]
            encoder = StaticEncoder().fit(static_dev)
            ds = build_window_dataset(view, tables.student_info, encoder=encoder,
                                      fit_static=False, days_per_week=cfg.days_per_week)
            mod = ds.keys["code_module"].to_numpy()
            test_mask = mod == m
            nontest = np.where(~test_mask)[0]
            rng = np.random.default_rng(SEED)
            val_idx = rng.choice(nontest, size=max(1, int(0.15 * len(nontest))), replace=False)
            val_mask = np.zeros(len(mod), bool); val_mask[val_idx] = True
            train_mask = ~test_mask & ~val_mask
            train, val, test = ds.subset(train_mask), ds.subset(val_mask), ds.subset(test_mask)
            if len(np.unique(test.y)) < 2 or len(np.unique(train.y)) < 2:
                continue

            # Symbolic thresholds on TRAIN ids only.
            train_ids = set(train.keys["id_student"])
            thr = RuleThresholds.fit(view, train_ids, days_per_week=cfg.days_per_week)
            sym_frame = SymbolicEngine(thr, days_per_week=cfg.days_per_week).apply(view).to_frame()
            sym_val = align_symbolic(sym_frame, val.keys)
            sym_test = align_symbolic(sym_frame, test.keys)

            gru = GRUModel(seed=SEED, arch="gru").fit(train, val)
            xgb = XGBoostFull(seed=SEED).fit(train, val)
            p_val_gru = gru.predict_proba(val)
            fusion = LogisticFusion(seed=SEED).fit(p_val_gru, sym_val, val.y)

            p_gru = gru.predict_proba(test)
            p_ns = fusion.predict_proba(p_gru, sym_test)
            iso = IsotonicRegression(out_of_bounds="clip").fit(fusion.predict_proba(p_val_gru, sym_val), val.y)
            p_nsc = iso.predict(p_ns)
            p_xgb = xgb.predict_proba(test)
            p_rules = sym_test.p_symbolic

            r = dict(window=k, held_out_module=m, n_test=len(test.y), base_rate=float(test.y.mean()),
                     auc_GRU=auc(test.y, p_gru), auc_NSGRU=auc(test.y, p_ns),
                     auc_NSGRUcal=auc(test.y, p_nsc), auc_XGBoost=auc(test.y, p_xgb),
                     auc_Rules=auc(test.y, p_rules))
            r["d_NSGRU_vs_GRU"] = r["auc_NSGRU"] - r["auc_GRU"]
            r["d_NSGRU_vs_XGB"] = r["auc_NSGRU"] - r["auc_XGBoost"]
            rows.append(r)
            print(f"W{k} holdout={m}: GRU={r['auc_GRU']:.4f} NS-GRU={r['auc_NSGRU']:.4f} "
                  f"XGB={r['auc_XGBoost']:.4f}  dNS-GRU(vsGRU)={r['d_NSGRU_vs_GRU']:+.4f} "
                  f"(vsXGB)={r['d_NSGRU_vs_XGB']:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).resolve().parents[1] / "outputs" / "coldstart.csv", index=False)
    print("\n== Mean over held-out modules (cold-start) ==")
    g = df.groupby("window")[["auc_GRU", "auc_NSGRU", "auc_NSGRUcal", "auc_XGBoost",
                              "d_NSGRU_vs_GRU", "d_NSGRU_vs_XGB"]].mean().round(4)
    print(g.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
