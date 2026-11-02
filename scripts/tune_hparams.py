"""Tuning hyperparameter sekali per window (validation PR-AUC) -> outputs/hparams.json.

GRU (juga dipakai Transformer) dan XGBoost. Dijalankan pada data penuh, satu seed.
run_full.py akan memuat hparams.json bila ada.

    python scripts/tune_hparams.py --windows 2 4 6 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from nsoulad.config import load_config
from nsoulad.data.loader import load_oulad
from nsoulad.data.windowing import build_window_view
from nsoulad.features.build import split_window_datasets
from nsoulad.models.tuning import tune_gru, tune_xgboost


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(cfg.data_dir))
    ap.add_argument("--windows", type=int, nargs="+", default=list(cfg.windows))
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    tables = load_oulad(args.data_dir)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    hp = {}
    for k in args.windows:
        view = build_window_view(tables, k, days_per_week=cfg.days_per_week)
        parts = split_window_datasets(view, tables.student_info, ratios=cfg.split.ratios,
                                      seed=args.seed, days_per_week=cfg.days_per_week)
        g = tune_gru(parts["train"], parts["val"], seed=args.seed)
        x = tune_xgboost(parts["train"], parts["val"], seed=args.seed)
        hp[str(k)] = {"gru": g["best"], "transformer": g["best"], "xgboost": x["best"]}
        print(f"Window {k}: GRU {g['best']} (PR-AUC {g['score']}) | XGB {x['best']} (PR-AUC {x['score']})")

    (out / "hparams.json").write_text(json.dumps(hp, indent=2), encoding="utf-8")
    print(f"\nDisimpan ke {out/'hparams.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
