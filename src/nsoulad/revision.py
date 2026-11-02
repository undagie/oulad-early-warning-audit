"""Pipeline revisi (SOIC 4706): satu panggilan = satu (kondisi, window, seed).

Berbeda dari pipeline.run_window, modul ini:
  - men-tuning GRU, LSTM, Transformer, XGBoost, dan Static-only PER SEED pada split
    validasi seed itu sendiri, dengan grid yang sama untuk ketiga encoder sekuensial
    (nested per-seed selection, budget tuning setara);
  - mendukung kondisi sensitivitas: leaky, lag visibilitas skor, cohort tetap,
    split presentation identity-disjoint, dan leave-one-module-out;
  - mengembalikan prediksi PER BARIS (val + test) beserta label, jenis outcome,
    atribut demografis, aktivasi & input mentah rule, sehingga semua analisis revisi
    (inferensi per mahasiswa, ekuivalensi, kalibrasi, fairness, operating point,
    validasi rule) dihitung ulang dari cache tanpa melatih ulang.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from nsoulad.config import Config
from nsoulad.data.loader import OuladTables
from nsoulad.data.schema import UNIT_KEYS
from nsoulad.data.windowing import build_window_view
from nsoulad.features.assessment import assessment_features
from nsoulad.features.build import (
    module_holdout_datasets,
    presentation_split_datasets,
    split_window_datasets,
)
from nsoulad.imbalance import class_pos_weight
from nsoulad.models import GRUModel, LogisticFusion, StaticOnly, XGBoostFull, align_symbolic
from nsoulad.models.util import static_features
from nsoulad.symbolic import RuleThresholds, SymbolicEngine
from nsoulad.symbolic.rules import RULE_IDS

_KEYS = list(UNIT_KEYS)

# Grid yang sama untuk ketiga encoder sekuensial (budget tuning setara).
NEURAL_GRID = [dict(hidden=h, dropout=0.3, lr=l) for h in (16, 32, 64) for l in (1e-3, 5e-4)]
XGB_GRID = [dict(max_depth=d, learning_rate=l) for d in (4, 5, 6) for l in (0.03, 0.05, 0.1)]
DEMOGRAPHICS = ["gender", "region", "age_band", "imd_band", "disability", "highest_education",
                "num_of_prev_attempts", "studied_credits"]
RULE_INPUTS = ["run_zero_active", "run_below_q1", "scheduled_resource_ratio", "first_score",
               "n_scored", "has_assessment", "n_late", "num_of_prev_attempts", "early_engagement",
               "early_median_course", "has_due_assessment", "active_days_7d_pre_deadline"]


@dataclass(frozen=True)
class Condition:
    name: str
    leaky: bool = False
    score_lag_days: int = 0
    fixed_cohort_weeks: tuple[int, int] | None = None
    include_withdrawn: bool = False  # cohort saat enrolment (termasuk withdrawal pra-cutoff)
    split: str = "student"          # student | presentation_disjoint | lomo
    module: str | None = None       # untuk split="lomo"


CONDITIONS = {
    "main": Condition("main"),
    "leaky": Condition("leaky", leaky=True),
    "lag7": Condition("lag7", score_lag_days=7),
    "lag14": Condition("lag14", score_lag_days=14),
    "lag21": Condition("lag21", score_lag_days=21),
    "fixed_cohort": Condition("fixed_cohort", fixed_cohort_weeks=(2, 8)),
    "enrolment_cohort": Condition("enrolment_cohort", include_withdrawn=True),
    "transfer_disjoint": Condition("transfer_disjoint", split="presentation_disjoint"),
}


def lomo_condition(module: str) -> Condition:
    return Condition(f"lomo_{module}", split="lomo", module=module)


def _prauc(y, p) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else 0.0


def _build_parts(tables: OuladTables, view, cond: Condition, seed: int, cfg: Config):
    if cond.split == "student":
        return split_window_datasets(view, tables.student_info, ratios=cfg.split.ratios, seed=seed,
                                     days_per_week=cfg.days_per_week)
    if cond.split == "presentation_disjoint":
        return presentation_split_datasets(
            view, tables.student_info, train_presentations=cfg.split.train_presentations,
            test_presentations=cfg.split.test_presentations, seed=seed,
            days_per_week=cfg.days_per_week, exclude_returning=True)
    if cond.split == "lomo":
        return module_holdout_datasets(view, tables.student_info, cond.module, seed=seed,
                                       days_per_week=cfg.days_per_week)
    raise ValueError(f"split tak dikenal: {cond.split}")


def _impute_pending_scores(parts: dict, view) -> dict[str, np.ndarray]:
    """Submission tanpa skor terlihat -> first/mean score = rata-rata train (bukan 0)."""
    aux = assessment_features(view)[_KEYS + ["score_pending", "n_scored"]]
    pending = {}
    for name, ds in parts.items():
        a = ds.keys.merge(aux, on=_KEYS, how="left")
        pending[name] = a["score_pending"].fillna(0).to_numpy(bool)
    tr = parts["train"]
    scored = tr.keys.merge(aux, on=_KEYS, how="left")["n_scored"].fillna(0).to_numpy() > 0
    # Bila belum ada skor terlihat sama sekali di train (lag panjang di window awal), fitur
    # skor tidak informatif; isi 0 seperti mahasiswa tanpa submission.
    fill = {c: float(tr.X_tab.loc[scored, c].mean()) if scored.any() else 0.0
            for c in ("first_score", "mean_score")}
    for name, ds in parts.items():
        if pending[name].any():
            ds.X_tab = ds.X_tab.copy()
            for c, v in fill.items():
                ds.X_tab.loc[pending[name], c] = v
    return pending


def _select(kind: str, grid: list[dict], fit_predict, val_y) -> tuple[dict, np.ndarray, np.ndarray, list]:
    """Latih tiap konfigurasi, pilih PR-AUC validasi tertinggi; kembalikan prediksi terpilih."""
    best, log = None, []
    for kw in grid:
        t0 = time.time()
        pv, pt = fit_predict(kw)
        s = _prauc(val_y, pv)
        log.append({"model": kind, **kw, "val_pr_auc": s, "seconds": round(time.time() - t0, 2)})
        if best is None or s > best[0]:
            best = (s, kw, pv, pt)
    return best[1], best[2], best[3], log


def run_condition(tables: OuladTables, cond: Condition, k: int, seed: int, cfg: Config,
                  archs: tuple[str, ...] = ("gru", "lstm", "transformer"),
                  hparams: dict | None = None, include_ra: bool = True,
                  view=None) -> tuple[pd.DataFrame, dict]:
    """Latih semua model untuk (kondisi, window, seed); kembalikan (prediksi per baris, meta).

    hparams: bila diberikan ({"gru": {...}, "xgboost": {...}, ...}), konfigurasi itu dipakai
    tanpa tuning (kondisi sensitivitas memakai pilihan per-seed dari kondisi utama).
    """
    t_start = time.time()
    if view is None:
        view = build_window_view(tables, k, days_per_week=cfg.days_per_week,
                                 feature_cutoff_day=10**6 if cond.leaky else None,
                                 score_lag_days=cond.score_lag_days,
                                 fixed_cohort_weeks=cond.fixed_cohort_weeks,
                                 include_withdrawn=cond.include_withdrawn)
    parts = _build_parts(tables, view, cond, seed, cfg)
    pending = _impute_pending_scores(parts, view)
    train, val, test = parts["train"], parts["val"], parts["test"]
    hp = hparams or {}

    # --- Symbolic (threshold dari train saja) ---
    thr = RuleThresholds.fit(view, set(train.keys["id_student"]), days_per_week=cfg.days_per_week)
    rule_res = SymbolicEngine(thr, days_per_week=cfg.days_per_week).apply(view)
    sym_frame = rule_res.to_frame()
    sym = {name: align_symbolic(sym_frame, p.keys) for name, p in parts.items()}
    rule_inputs = rule_res.rule_frame[_KEYS + RULE_INPUTS].drop_duplicates(_KEYS)

    preds = {"val": {}, "test": {}}
    tuning_log, selected = [], {}

    def keep(name, pv, pt):
        preds["val"][name], preds["test"][name] = pv, pt

    # --- XGBoost & Static-only (grid yang sama) ---
    for name, cls in (("XGBoost", XGBoostFull), ("Static-only", TunedStaticOnly)):
        key = "xgboost" if name == "XGBoost" else "static"
        grid = [hp[key]] if key in hp else XGB_GRID

        def fp(kw, cls=cls):
            m = cls(seed=seed, **kw).fit(train, val)
            return m.predict_proba(val), m.predict_proba(test)
        selected[key], pv, pt, log = _select(name, grid, fp, val.y)
        keep(name, pv, pt)
        tuning_log += log

    keep("Rules-only", sym["val"].p_symbolic, sym["test"].p_symbolic)

    # --- Encoder sekuensial (grid sama untuk GRU/LSTM/Transformer) ---
    label = {"gru": "GRU", "lstm": "LSTM", "transformer": "Transformer"}
    for arch in archs:
        grid = [hp[arch]] if arch in hp else NEURAL_GRID

        def fp(kw, arch=arch):
            m = GRUModel(seed=seed, arch=arch, **kw).fit(train, val)
            return m.predict_proba(val), m.predict_proba(test)
        selected[arch], pv, pt, log = _select(label[arch], grid, fp, val.y)
        keep(label[arch], pv, pt)
        tuning_log += log

    fusion_coef = {}
    if "gru" in archs:
        pv_gru, pt_gru = preds["val"]["GRU"], preds["test"]["GRU"]
        for fs, name in (("full", "NS-GRU"), ("compact", "NS-GRU-2in")):
            fus = LogisticFusion(seed=seed, feature_set=fs).fit(pv_gru, sym["val"], val.y)
            keep(name, fus.predict_proba(pv_gru, sym["val"]), fus.predict_proba(pt_gru, sym["test"]))
            fusion_coef[fs] = {"intercept": float(fus.clf.intercept_[0]), **fus.coefficients()}
        if include_ra:
            ra = GRUModel(seed=seed, arch="gru", **selected["gru"])
            ra.fit(train, val, train_extra=sym["train"].as_features(), val_extra=sym["val"].as_features())
            keep("NS-GRU-RA", ra.predict_proba(val, extra=sym["val"].as_features()),
                 ra.predict_proba(test, extra=sym["test"].as_features()))

    # --- Bingkai per baris ---
    info = tables.student_info[_KEYS + ["final_result"] + DEMOGRAPHICS]
    frames = []
    for part in ("val", "test"):
        ds = parts[part]
        df = ds.keys.copy()
        df["split"] = part
        df["y"] = ds.y
        df["score_pending"] = pending[part].astype(int)
        df = df.merge(info, on=_KEYS, how="left")
        for name, p in preds[part].items():
            df[f"p_{name}"] = p
        b = sym[part]
        df["p_symbolic"], df["rule_count"], df["high_severity_count"] = (
            b.p_symbolic, b.rule_count, b.high_severity_count)
        for r in RULE_IDS:
            df[r] = b.triggered[r].to_numpy()
        df = df.merge(rule_inputs, on=_KEYS, how="left", suffixes=("", "_rule"))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "seed", seed)
    out.insert(0, "window", k)
    out.insert(0, "condition", cond.name)

    meta = {
        "condition": asdict(cond), "window": k, "seed": seed,
        "n_train": train.n, "n_val": val.n, "n_test": test.n,
        "base_rate_test": float(test.y.mean()) if test.n else float("nan"),
        "selected": selected, "fusion_coef": fusion_coef, "tuning_log": tuning_log,
        "seconds": round(time.time() - t_start, 1),
    }
    return out, meta


class TunedStaticOnly(StaticOnly):
    """Static-only dengan kedalaman & learning rate yang dituning (grid = XGB_GRID)."""

    def __init__(self, seed: int = 42, max_depth: int = 4, learning_rate: float = 0.05):
        super().__init__(seed=seed)
        self.max_depth, self.learning_rate = max_depth, learning_rate

    def fit(self, train, val=None) -> "TunedStaticOnly":
        self.model = XGBClassifier(n_estimators=200, max_depth=self.max_depth,
                                   learning_rate=self.learning_rate, subsample=0.9,
                                   scale_pos_weight=class_pos_weight(train.y), eval_metric="aucpr",
                                   random_state=self.seed, n_jobs=0)
        self.model.fit(static_features(train), train.y)
        return self
