"""E4-SHAP: baseline post-hoc XAI (SHAP atas XGBoost) untuk perbandingan actionability.

Untuk mahasiswa yang diprediksi at-risk, ukur seberapa sering atribusi SHAP teratas
jatuh pada fitur immutable (demografi enrollment) -> tidak actionable. Pembanding:
rule-based NS-GRU = 0% (lihat counterfactual_comparison.csv).

    python scripts/run_shap.py --windows 4 8
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from nsoulad.config import load_config
from nsoulad.data.loader import load_oulad
from nsoulad.data.windowing import build_window_view
from nsoulad.evaluation.shap_baseline import shap_actionability
from nsoulad.features.build import split_window_datasets
from nsoulad.imbalance import tune_threshold
from nsoulad.models import XGBoostFull, full_features

warnings.filterwarnings("ignore")


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(cfg.data_dir))
    ap.add_argument("--windows", type=int, nargs="+", default=list(cfg.windows))
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    tables = load_oulad(args.data_dir)
    rows = []
    for k in args.windows:
        view = build_window_view(tables, k, days_per_week=cfg.days_per_week)
        parts = split_window_datasets(view, tables.student_info, ratios=cfg.split.ratios,
                                      seed=args.seed, days_per_week=cfg.days_per_week)
        train, val, test = parts["train"], parts["val"], parts["test"]
        xgb = XGBoostFull(seed=args.seed).fit(train)
        t = tune_threshold(val.y, xgb.predict_proba(val))
        pred = (xgb.predict_proba(test) >= t).astype(int)

        immutable = set(test.encoder.feature_names_)  # kolom demografi statis = immutable
        res = shap_actionability(xgb.model, full_features(test), immutable, pred, top_k=3)
        res["window"] = k
        rows.append(res)
        print(f"Window {k}: {res}")

    df = pd.DataFrame(rows)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "shap_actionability.csv", index=False)
    print("\n=== SHAP actionability (XGBoost) vs rule-based (0% by construction) ===")
    print(df.to_string(index=False))
    print(f"\nDisimpan ke {out/'shap_actionability.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
