"""Robustness PoC (pre-registered Track D2) — does NS-GRU degrade more GRACEFULLY than
GRU/XGBoost when the clickstream is corrupted?

Scenario: the VLE logging pipeline loses a fraction rho of interaction records
(realistic deployment failure), while the assessment/registration data (SIS) stays intact.
We TRAIN once on clean data, then at inference rebuild TEST features from a corrupted
studentVle. This degrades the GRU sequence, the course-structure tabular features, and the
clickstream-based rules (R1/R2/R7) consistently, while assessment features/rules survive.

Hypothesis (pre-registered): symbolic evidence (esp. assessment/deadline rules) buffers the
prediction, so NS-GRU's AUC drop < GRU's AUC drop as rho grows. PoC on 2 windows, seed 42;
if the buffer is real, expand to the full grid. Output: outputs/robustness.csv
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nsoulad.config import load_config
from nsoulad.data.loader import load_oulad
from nsoulad.data.windowing import build_window_view
from nsoulad.features.build import build_window_dataset, split_window_datasets
from nsoulad.models import GRUModel, LogisticFusion, XGBoostFull, align_symbolic
from nsoulad.symbolic import RuleThresholds, SymbolicEngine

WINDOWS = [2, 8]
RHOS = [0.0, 0.25, 0.5, 0.75, 0.9]
SEED = 42


def corrupt_vle(tables, rho: float, seed: int):
    if rho <= 0:
        return tables
    svle = tables.student_vle
    rng = np.random.default_rng(1000 + seed)
    keep = rng.random(len(svle)) >= rho
    return replace(tables, student_vle=svle.loc[keep].reset_index(drop=True))


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def main():
    cfg = load_config()
    tables = load_oulad(str(cfg.data_dir))
    rows = []

    for k in WINDOWS:
        # ---- Train ONCE on clean data ----
        view = build_window_view(tables, k, days_per_week=cfg.days_per_week)
        parts = split_window_datasets(view, tables.student_info, ratios=cfg.split.ratios,
                                      seed=SEED, days_per_week=cfg.days_per_week)
        train, val, test = parts["train"], parts["val"], parts["test"]
        test_ids = set(test.keys["id_student"])

        train_ids = set(train.keys["id_student"])
        thr = RuleThresholds.fit(view, train_ids, days_per_week=cfg.days_per_week)
        engine = SymbolicEngine(thr, days_per_week=cfg.days_per_week)
        sym_frame_clean = engine.apply(view).to_frame()
        sym_val = align_symbolic(sym_frame_clean, val.keys)

        gru = GRUModel(seed=SEED, arch="gru").fit(train, val)
        xgb = XGBoostFull(seed=SEED).fit(train, val)
        fusion = LogisticFusion(seed=SEED).fit(gru.predict_proba(val), sym_val, val.y)

        # ---- Evaluate under increasing corruption (rebuild TEST features each rho) ----
        for rho in RHOS:
            tinit = corrupt_vle(tables, rho, SEED)
            view_c = build_window_view(tinit, k, days_per_week=cfg.days_per_week)
            # Rebuild full cohort dataset with the CLEAN train-fit encoder, then subset to test.
            ds_c = build_window_dataset(view_c, tinit.student_info, encoder=train.encoder,
                                        fit_static=False, days_per_week=cfg.days_per_week)
            mask = np.isin(ds_c.keys["id_student"].to_numpy(), list(test_ids))
            test_c = ds_c.subset(mask)
            # Recompute symbolic on corrupted view (thresholds from clean train), align to test_c.
            sym_c_frame = engine.apply(view_c).to_frame()
            sym_c_test = align_symbolic(sym_c_frame, test_c.keys)

            y = test_c.y
            p_gru = gru.predict_proba(test_c)
            p_ns = fusion.predict_proba(p_gru, sym_c_test)
            p_xgb = xgb.predict_proba(test_c)
            p_rules = sym_c_test.p_symbolic
            rows.append(dict(window=k, rho=rho, n=len(y), base_rate=float(y.mean()),
                             auc_GRU=auc(y, p_gru), auc_NSGRU=auc(y, p_ns),
                             auc_XGBoost=auc(y, p_xgb), auc_Rules=auc(y, p_rules)))
            print(f"W{k} rho={rho:>4}: GRU={rows[-1]['auc_GRU']:.4f}  "
                  f"NS-GRU={rows[-1]['auc_NSGRU']:.4f}  XGB={rows[-1]['auc_XGBoost']:.4f}  "
                  f"Rules={rows[-1]['auc_Rules']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).resolve().parents[1] / "outputs" / "robustness.csv", index=False)

    # ---- Degradation summary: AUC drop from rho=0, GRU vs NS-GRU ----
    print("\n-- AUC drop from clean (rho=0); does NS-GRU buffer vs GRU? --")
    for k in WINDOWS:
        d = df[df.window == k].set_index("rho")
        base = d.loc[0.0]
        for rho in RHOS[1:]:
            dg = base.auc_GRU - d.loc[rho].auc_GRU
            dn = base.auc_NSGRU - d.loc[rho].auc_NSGRU
            print(f"W{k} rho={rho}: dropGRU={dg:+.4f}  dropNSGRU={dn:+.4f}  "
                  f"buffer(NS better by)={dg - dn:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
