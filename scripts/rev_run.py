"""Revisi SOIC 4706: latih & simpan prediksi per baris untuk satu kondisi.

Hasil per (kondisi, window, seed) disimpan ke outputs/revision/preds/<kondisi>/ sebagai
parquet (prediksi val+test per baris) dan json (hyperparameter terpilih, log tuning,
koefisien fusion). Resume otomatis: file yang sudah ada dilewati.

    python scripts/rev_run.py --condition main                 # tuning per seed, semua model
    python scripts/rev_run.py --condition lag14                # pakai hparam terpilih kondisi main
    python scripts/rev_run.py --condition lomo --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from nsoulad.config import load_config
from nsoulad.data.loader import load_oulad
from nsoulad.data.windowing import build_window_view
from nsoulad.revision import CONDITIONS, lomo_condition, run_condition

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "revision" / "preds"
MODULES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]


def _main_hparams(k: int, seed: int) -> dict:
    meta = json.loads((OUT / "main" / f"w{k}_s{seed}.json").read_text(encoding="utf-8"))
    sel = meta["selected"]
    return {"gru": sel["gru"], "xgboost": sel["xgboost"], "static": sel["static"]}


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=[*CONDITIONS, "lomo"])
    ap.add_argument("--windows", type=int, nargs="+", default=list(cfg.windows))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    tables = load_oulad(str(ROOT / cfg.data_dir))
    is_main = args.condition == "main"
    tune = args.condition in ("main", "transfer_disjoint")
    archs = ("gru", "lstm", "transformer") if is_main else ("gru",)
    conds = [lomo_condition(m) for m in MODULES] if args.condition == "lomo" else [CONDITIONS[args.condition]]

    for k in args.windows:
        base = conds[0]
        view = build_window_view(tables, k, days_per_week=cfg.days_per_week,
                                 feature_cutoff_day=10**6 if base.leaky else None,
                                 score_lag_days=base.score_lag_days,
                                 fixed_cohort_weeks=base.fixed_cohort_weeks,
                                 include_withdrawn=base.include_withdrawn)
        for cond in conds:
            d = OUT / cond.name
            d.mkdir(parents=True, exist_ok=True)
            for s in args.seeds:
                stem = d / f"w{k}_s{s}"
                if stem.with_suffix(".parquet").exists():
                    continue
                t0 = time.time()
                hp = None if tune else _main_hparams(k, s)
                preds, meta = run_condition(tables, cond, k, s, cfg, archs=archs, hparams=hp,
                                            include_ra=is_main, view=view)
                preds.to_parquet(stem.with_suffix(".parquet"), index=False)
                stem.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
                print(f"{cond.name} w{k} s{s}: n_test={meta['n_test']} "
                      f"selected={meta['selected']} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
