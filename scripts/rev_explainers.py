"""Revisi SOIC 4706: kontras struktural rule-based alert vs penjelas post-hoc.

Bagian A — atribusi grouped Shapley EKSAK pada model fusion yang dilaporkan (NS-GRU 2-input)
  di Week 2/4/6/8. Enam kelompok input: sekuens engagement, asesmen, struktur course,
  atribut personal (demografi + riwayat), konteks course (modul/presentation), dan ringkasan
  rule (P_symbolic). v(S) = rata-rata output fusion saat kelompok di luar S diganti nilai
  latar dari sampel train (interventional). Dengan 6 kelompok, 64 koalisi dihitung eksak.
Bagian B — TreeSHAP XGBoost dijumlahkan per kelompok yang sama (pembanding lama, kini grouped
  dan pada empat minggu).
Bagian C — DiCE pada model tabular interpretable: tanpa batasan vs dibatasi ke fitur perilaku
  yang dapat diubah; kegagalan generate dicatat beserta karakteristik kasusnya.

    python scripts/rev_explainers.py --seed 42 --n-cases 300 --n-dice 100
"""

from __future__ import annotations

import argparse
import json
import warnings
from itertools import product
from math import factorial
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch

from nsoulad.config import load_config
from nsoulad.data.loader import load_oulad
from nsoulad.data.windowing import build_window_view
from nsoulad.features.assessment import ASSESSMENT_FEATURES
from nsoulad.features.course_structure import COURSE_STRUCTURE_FEATURES
from nsoulad.features.build import split_window_datasets
from nsoulad.imbalance import tune_threshold
from nsoulad.models import GRUModel, LogisticFusion, XGBoostFull, align_symbolic
from nsoulad.models.util import full_features
from nsoulad.revision import _impute_pending_scores
from nsoulad.symbolic import RuleThresholds, SymbolicEngine

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "revision" / "tables"
GROUPS = ["engagement_sequence", "assessment", "course_structure", "personal_attributes",
          "course_context", "rule_summary"]


def _tab_groups(columns: list[str]) -> dict[str, list[int]]:
    ctx = [i for i, c in enumerate(columns) if c.startswith(("code_module=", "code_presentation="))]
    asm = [columns.index(c) for c in ASSESSMENT_FEATURES]
    crs = [columns.index(c) for c in COURSE_STRUCTURE_FEATURES]
    personal = [i for i in range(len(columns)) if i not in set(ctx) | set(asm) | set(crs)]
    return {"assessment": asm, "course_structure": crs, "personal_attributes": personal, "course_context": ctx}


def _fused(gru: GRUModel, fus: LogisticFusion, seq: np.ndarray, tab: np.ndarray, psym: np.ndarray) -> np.ndarray:
    xs = torch.tensor((seq - gru.seq_mean_) / gru.seq_std_, dtype=torch.float32, device=gru.device)
    xt = torch.tensor((tab - gru.tab_mean_) / gru.tab_std_, dtype=torch.float32, device=gru.device)
    gru.net.eval()
    with torch.no_grad():
        pn = torch.sigmoid(gru.net(xs, xt)).cpu().numpy()
    X = fus.scaler.transform(np.column_stack([pn, psym]))
    return fus.clf.predict_proba(X)[:, 1]


def grouped_shapley(gru, fus, seq, tab, psym, bg_idx, case_idx, tab_groups) -> np.ndarray:
    """Shapley eksak per kelompok untuk tiap kasus; returns [n_cases, 6]."""
    G = len(GROUPS)
    masks = np.array(list(product([0, 1], repeat=G)), dtype=bool)          # [64, G]
    B = len(bg_idx)
    phi = np.zeros((len(case_idx), G))
    weight = {s: factorial(s) * factorial(G - s - 1) / factorial(G) for s in range(G)}
    index = {tuple(m): j for j, m in enumerate(masks)}
    full = index[(True,) * G]
    max_gap = 0.0
    for n, i in enumerate(case_idx):
        S = np.repeat(masks, B, axis=0)                                       # [64B, G]
        bg = np.tile(bg_idx, len(masks))
        q = np.where(S[:, 0, None, None], seq[i][None], seq[bg])
        t = tab[bg].copy()
        for g, name in enumerate(GROUPS[1:5], start=1):
            cols = tab_groups[name]
            t[np.ix_(S[:, g], cols)] = tab[i, cols]
        ps = np.where(S[:, 5], psym[i], psym[bg])
        v = _fused(gru, fus, q, t, ps).reshape(len(masks), B).mean(axis=1)    # v(S)
        for g in range(G):
            for j, m in enumerate(masks):
                if m[g]:
                    continue
                with_g = m.copy()
                with_g[g] = True
                phi[n, g] += weight[int(m.sum())] * (v[index[tuple(with_g)]] - v[j])
        # Efisiensi Shapley: jumlah atribusi = v(semua) - v(kosong).
        max_gap = max(max_gap, abs(phi[n].sum() - (v[full] - v[0])))
    print(f"  max |sum(phi) - (v(all) - v(empty))| = {max_gap:.2e}", flush=True)
    return phi


def _summary(phi: np.ndarray, label: str, k: int) -> list[dict]:
    absphi = np.abs(phi)
    share = absphi / np.maximum(absphi.sum(axis=1, keepdims=True), 1e-12)
    rank = np.argsort(-absphi, axis=1)
    rows = []
    for g, name in enumerate(GROUPS):
        rows.append({"explainer": label, "window": k, "group": name, "n_cases": len(phi),
                     "mean_abs_share": float(share[:, g].mean()),
                     "mean_signed": float(phi[:, g].mean()),
                     "top1_rate": float((rank[:, 0] == g).mean()),
                     "top2_rate": float((rank[:, :2] == g).any(axis=1).mean())})
    return rows


def treeshap_grouped(xgb: XGBoostFull, test, columns_flat: list[str], case_idx, psym_unused=None) -> np.ndarray:
    import shap

    X = full_features(test).iloc[case_idx]
    sv = shap.TreeExplainer(xgb.model).shap_values(X)
    cols = list(X.columns)
    seq_cols = [i for i, c in enumerate(cols) if c not in set(test.X_tab.columns)]
    tg = _tab_groups(list(test.X_tab.columns))
    offset = len(cols) - test.X_tab.shape[1]
    out = np.zeros((len(case_idx), len(GROUPS)))
    out[:, 0] = sv[:, seq_cols].sum(axis=1)
    for g, name in enumerate(GROUPS[1:5], start=1):
        out[:, g] = sv[:, [offset + i for i in tg[name]]].sum(axis=1)
    return out  # rule_summary tidak dipakai XGBoost (kolom tetap 0)


def run_shapley(tables, cfg, seed: int, n_cases: int, n_bg: int) -> pd.DataFrame:
    rows = []
    for k in cfg.windows:
        meta = json.loads((ROOT / "outputs" / "revision" / "preds" / "main" / f"w{k}_s{seed}.json")
                          .read_text(encoding="utf-8"))
        view = build_window_view(tables, k, days_per_week=cfg.days_per_week)
        parts = split_window_datasets(view, tables.student_info, ratios=cfg.split.ratios, seed=seed,
                                      days_per_week=cfg.days_per_week)
        _impute_pending_scores(parts, view)
        train, val, test = parts["train"], parts["val"], parts["test"]
        thr = RuleThresholds.fit(view, set(train.keys["id_student"]), days_per_week=cfg.days_per_week)
        sf = SymbolicEngine(thr, days_per_week=cfg.days_per_week).apply(view).to_frame()
        s_va, s_te, s_tr = (align_symbolic(sf, d.keys) for d in (val, test, train))
        gru = GRUModel(seed=seed, arch="gru", **meta["selected"]["gru"]).fit(train, val)
        fus = LogisticFusion(seed=seed, feature_set="compact").fit(gru.predict_proba(val), s_va, val.y)
        xgb = XGBoostFull(seed=seed, **meta["selected"]["xgboost"]).fit(train)

        p_val = fus.predict_proba(gru.predict_proba(val), s_va)
        p_test = fus.predict_proba(gru.predict_proba(test), s_te)
        flagged = np.where(p_test >= tune_threshold(val.y, p_val))[0]
        rng = np.random.default_rng(seed + k)
        case_idx = rng.choice(flagged, size=min(n_cases, len(flagged)), replace=False)

        seq = np.concatenate([train.X_seq, test.X_seq])
        tab = np.concatenate([train.X_tab.to_numpy(float), test.X_tab.to_numpy(float)])
        psym = np.concatenate([s_tr.p_symbolic, s_te.p_symbolic])
        bg_idx = rng.choice(train.n, size=n_bg, replace=False)
        phi = grouped_shapley(gru, fus, seq, tab, psym, bg_idx, train.n + case_idx,
                              _tab_groups(list(train.X_tab.columns)))
        rows += _summary(phi, "Grouped Shapley (fused NS-GRU)", k)
        rows += _summary(treeshap_grouped(xgb, test, None, case_idx), "Grouped TreeSHAP (XGBoost)", k)
        print(f"W{k}: {len(case_idx)} flagged cases explained", flush=True)
    return pd.DataFrame(rows)


def run_dice(tables, cfg, seed: int, n_cases: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    import dice_ml
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    import run_counterfactual as rc  # frame interpretable yang sama dengan analisis awal

    from nsoulad.data.labels import make_binary_label
    from nsoulad.data.splits import student_level_split

    mutable = ["total_clicks", "total_active_days", "n_submitted", "mean_score", "first_score", "n_late",
               "scheduled_resource_ratio", "active_days_7d_pre_deadline"]
    summary, cases = [], []
    for k in cfg.windows:
        view = build_window_view(tables, k, days_per_week=cfg.days_per_week)
        frame = rc._interpretable_frame(view).merge(
            make_binary_label(tables.student_info)[rc._KEYS + ["at_risk"]], on=rc._KEYS, how="left")
        for c in rc.CONTINUOUS:
            frame[c] = pd.to_numeric(frame[c], errors="coerce").astype(float).fillna(0.0)
        feat = [c for c in frame.columns if c not in rc._KEYS + ["at_risk"]]
        cat = [c for c in feat if c not in rc.CONTINUOUS]
        for c in cat:
            frame[c] = frame[c].astype("object").fillna("missing").astype(str)
        split = student_level_split(frame[rc._KEYS + ["at_risk"]], ratios=cfg.split.ratios, seed=seed)
        tr = frame[frame.id_student.isin(set(split.train.id_student))]
        te = frame[frame.id_student.isin(set(split.test.id_student))]
        clf = Pipeline([("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                                                    ("num", "passthrough", rc.CONTINUOUS)])),
                        ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed, n_jobs=-1))])
        clf.fit(tr[feat], tr.at_risk)
        exp = dice_ml.Dice(dice_ml.Data(dataframe=tr[feat + ["at_risk"]], continuous_features=rc.CONTINUOUS,
                                        outcome_name="at_risk"),
                           dice_ml.Model(model=clf, backend="sklearn"), method="random")
        # Recourse hanya bermakna untuk kasus at-risk yang memang ditandai model (prob >= 0.5).
        pos = te[te.at_risk == 1]
        pos = pos[clf.predict_proba(pos[feat])[:, 1] >= 0.5]
        queries = pos.sample(n=min(n_cases, len(pos)), random_state=seed)
        for mode, vary in (("unconstrained", "all"), ("mutable-only", mutable)):
            changed, failures = [], 0
            for _, row in queries.iterrows():
                q = row[feat].to_frame().T
                risk = float(clf.predict_proba(q)[0, 1])
                rec = {"window": k, "mode": mode, "risk": risk, "model_flags_at_risk": risk >= 0.5}
                try:
                    cf = exp.generate_counterfactuals(q, total_CFs=1, desired_class=0, features_to_vary=vary,
                                                      random_seed=seed)
                    cf_df = cf.cf_examples_list[0].final_cfs_df
                    if cf_df is None or len(cf_df) == 0:
                        raise RuntimeError("no counterfactual within search budget")
                    diff = {c for c in feat if str(row[c]) != str(cf_df.iloc[0][c])}
                    changed.append(diff)
                    rec.update(success=True, n_changed=len(diff), immutable_changed=bool(diff & rc.IMMUTABLE),
                               reason="")
                except Exception as e:  # noqa: BLE001 — kegagalan dicatat, bukan dibuang
                    failures += 1
                    rec.update(success=False, n_changed=np.nan, immutable_changed=np.nan,
                               reason=type(e).__name__ + ": " + str(e)[:80])
                cases.append(rec)
            ok = [c for c in changed]
            summary.append({"window": k, "mode": mode, "attempted": len(queries), "succeeded": len(ok),
                            "failed": failures,
                            "immutable_change_rate": float(np.mean([bool(c & rc.IMMUTABLE) for c in ok])) if ok else np.nan,
                            "mean_features_changed": float(np.mean([len(c) for c in ok])) if ok else np.nan})
            print(f"DiCE W{k} {mode}: {len(ok)}/{len(queries)} succeeded", flush=True)
    return pd.DataFrame(summary), pd.DataFrame(cases)


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-cases", type=int, default=300)
    ap.add_argument("--n-bg", type=int, default=64)
    ap.add_argument("--n-dice", type=int, default=100)
    ap.add_argument("--skip-shapley", action="store_true")
    ap.add_argument("--skip-dice", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tables = load_oulad(str(ROOT / cfg.data_dir))
    if not args.skip_shapley:
        res = run_shapley(tables, cfg, args.seed, args.n_cases, args.n_bg)
        res.to_csv(OUT / "explainer_grouped_attribution.csv", index=False)
        print(res.round(3).to_string(index=False))
    if not args.skip_dice:
        summ, cases = run_dice(tables, cfg, args.seed, args.n_dice)
        summ.to_csv(OUT / "explainer_dice_summary.csv", index=False)
        cases.to_csv(OUT / "explainer_dice_cases.csv", index=False)
        print(summ.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
