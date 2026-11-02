"""Uji ulang klaim reliance fusion dengan fusion satu-fitur-per-cabang.

Fusion "full" memakai tiga fitur symbolic kolinear (P_symbolic, rule_count,
high_severity_count), sehingga koefisien individualnya tidak stabil. Skrip ini:

  A. Data penuh (Week 2/4/6/8 x seed, HP Tabel 1):
     - diagnosis kolinearitas fitur full (korelasi & VIF di val);
     - koefisien full (mentah) vs compact [P_neural, P_symbolic] terstandarisasi;
     - drop-column: AUC test GRU (= fusion tanpa symbolic; logistik satu fitur monoton)
       vs fusion full vs fusion compact.
  B. Kurva fraksi data (Week 2, 1-100%, GRU default seperti E3/run_fusion_analysis.py):
     rasio reliance compact |b_sym|/|b_neural| dan dAUC compact - GRU.
     Catatan: fusion tetap di-fit pada val PENUH (sama seperti E3), jadi bagian B
     mewarisi confound label-validasi yang sama.

    python scripts/run_fusion_reliance_check.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from nsoulad.config import load_config  # noqa: E402
from nsoulad.data.loader import load_oulad  # noqa: E402
from nsoulad.data.windowing import build_window_view  # noqa: E402
from nsoulad.features.build import split_window_datasets  # noqa: E402
from nsoulad.models import GRUModel, LogisticFusion, align_symbolic  # noqa: E402
from nsoulad.models.fusion import FUSION_FEATURES  # noqa: E402
from nsoulad.symbolic import RuleThresholds, SymbolicEngine  # noqa: E402


def _stratified_mask(y, frac, seed):
    if frac >= 1.0:
        return np.ones(len(y), dtype=bool)
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(y), dtype=bool)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        mask[idx[: max(2, int(round(len(idx) * frac)))]] = True
    return mask


def _vif(X: np.ndarray) -> np.ndarray:
    out = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        r2 = LinearRegression().fit(others, X[:, j]).score(others, X[:, j])
        out.append(np.inf if r2 >= 1 else 1.0 / (1.0 - r2))
    return np.array(out)


def _fit_all(view, parts, tr, seed, cfg, gru_hp):
    val, test = parts["val"], parts["test"]
    th = RuleThresholds.fit(view, set(tr.keys["id_student"]), days_per_week=cfg.days_per_week)
    sf = SymbolicEngine(th, days_per_week=cfg.days_per_week).apply(view).to_frame()
    sym_val, sym_test = align_symbolic(sf, val.keys), align_symbolic(sf, test.keys)
    gru = GRUModel(seed=seed, arch="gru", **gru_hp).fit(tr, val)
    pv, pt = gru.predict_proba(val), gru.predict_proba(test)
    full = LogisticFusion(seed=seed, feature_set="full").fit(pv, sym_val, val.y)
    comp = LogisticFusion(seed=seed, feature_set="compact").fit(pv, sym_val, val.y)
    return dict(
        X_val=sym_val.fusion_matrix(pv), full=full, comp=comp,
        auc_gru=roc_auc_score(test.y, pt),
        auc_full=roc_auc_score(test.y, full.predict_proba(pt, sym_test)),
        auc_comp=roc_auc_score(test.y, comp.predict_proba(pt, sym_test)),
    )


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(cfg.data_dir))
    ap.add_argument("--windows", type=int, nargs="+", default=list(cfg.windows))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--frac-window", type=int, default=2)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.25, 1.0])
    ap.add_argument("--frac-seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out = Path(args.out)
    hp_all = json.loads((out / "hparams.json").read_text(encoding="utf-8"))
    tables = load_oulad(args.data_dir)

    # --- A. Data penuh ---
    rows_a = []
    for k in args.windows:
        view = build_window_view(tables, k, days_per_week=cfg.days_per_week)
        gru_hp = hp_all.get(str(k), {}).get("gru", {})
        for s in args.seeds:
            parts = split_window_datasets(view, tables.student_info, ratios=cfg.split.ratios,
                                          seed=s, days_per_week=cfg.days_per_week)
            r = _fit_all(view, parts, parts["train"], s, cfg, gru_hp)
            X = r["X_val"]
            corr = np.corrcoef(X[:, 1:], rowvar=False)
            vif = _vif(X)
            row = {"window": k, "seed": s,
                   "corr_psym_rulecount": corr[0, 1], "corr_psym_highsev": corr[0, 2],
                   "corr_rulecount_highsev": corr[1, 2],
                   **{f"vif_{f}": v for f, v in zip(FUSION_FEATURES, vif)},
                   **{f"full_coef_{f}": v for f, v in r["full"].coefficients().items()},
                   **{f"compact_coef_{f}": v for f, v in r["comp"].coefficients().items()},
                   "auc_gru": r["auc_gru"], "auc_full": r["auc_full"], "auc_compact": r["auc_comp"]}
            row["compact_reliance_ratio"] = abs(row["compact_coef_P_symbolic"]) / abs(
                row["compact_coef_P_neural"])
            rows_a.append(row)
            print(f"[A] window {k} seed {s}: b_sym(compact)={row['compact_coef_P_symbolic']:+.3f} "
                  f"dAUC compact-GRU={r['auc_comp'] - r['auc_gru']:+.4f}", flush=True)

    a = pd.DataFrame(rows_a)
    a["dauc_full_minus_gru"] = a["auc_full"] - a["auc_gru"]
    a["dauc_compact_minus_gru"] = a["auc_compact"] - a["auc_gru"]
    a.to_csv(out / "fusion_reliance_check_full.csv", index=False)

    cols = ["full_coef_P_symbolic", "full_coef_rule_count", "full_coef_high_severity_count",
            "compact_coef_P_neural", "compact_coef_P_symbolic", "compact_reliance_ratio",
            "dauc_full_minus_gru", "dauc_compact_minus_gru",
            "corr_psym_rulecount", "vif_P_symbolic", "vif_rule_count"]
    sa = a.groupby("window")[cols].agg(["mean", "std"])
    sa.columns = [f"{c}_{s}" for c, s in sa.columns]
    sign = a.groupby("window").agg(
        n_seeds=("seed", "nunique"),
        full_psym_sign_flips=("full_coef_P_symbolic", lambda v: int((v > 0).sum() * (v < 0).sum() > 0)),
        compact_psym_n_positive=("compact_coef_P_symbolic", lambda v: int((v > 0).sum())),
        compact_dauc_n_positive=("dauc_compact_minus_gru", lambda v: int((v > 0).sum())),
    )
    sa = sa.join(sign).reset_index()
    sa.to_csv(out / "fusion_reliance_check_full_summary.csv", index=False)

    # --- B. Kurva fraksi data (Week frac-window, GRU default seperti E3) ---
    rows_b = []
    view = build_window_view(tables, args.frac_window, days_per_week=cfg.days_per_week)
    for s in args.frac_seeds:
        parts = split_window_datasets(view, tables.student_info, ratios=cfg.split.ratios,
                                      seed=s, days_per_week=cfg.days_per_week)
        for frac in args.fractions:
            tr = parts["train"].subset(_stratified_mask(parts["train"].y, frac, s))
            r = _fit_all(view, parts, tr, s, cfg, {})
            cc = r["comp"].coefficients()
            rows_b.append({"seed": s, "frac": frac, "n_train": tr.n,
                           "compact_coef_P_neural": cc["P_neural"],
                           "compact_coef_P_symbolic": cc["P_symbolic"],
                           "compact_reliance_ratio": abs(cc["P_symbolic"]) / abs(cc["P_neural"]),
                           "auc_gru": r["auc_gru"], "auc_full": r["auc_full"],
                           "auc_compact": r["auc_comp"]})
            print(f"[B] seed {s} frac {frac:.2f}: ratio={rows_b[-1]['compact_reliance_ratio']:.3f}",
                  flush=True)
    b = pd.DataFrame(rows_b)
    b["dauc_compact_minus_gru"] = b["auc_compact"] - b["auc_gru"]
    b["dauc_full_minus_gru"] = b["auc_full"] - b["auc_gru"]
    b.to_csv(out / "fusion_reliance_check_frac.csv", index=False)
    sb = b.groupby("frac")[["compact_reliance_ratio", "compact_coef_P_symbolic",
                            "dauc_compact_minus_gru", "dauc_full_minus_gru"]].agg(["mean", "std"])
    sb.columns = [f"{c}_{s}" for c, s in sb.columns]
    sb.reset_index().to_csv(out / "fusion_reliance_check_frac_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== A. Data penuh (mean/std atas seed) ===")
    print(sa.round(4).T.to_string())
    print("\n=== B. Kurva fraksi (Week %d) ===" % args.frac_window)
    print(sb.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
