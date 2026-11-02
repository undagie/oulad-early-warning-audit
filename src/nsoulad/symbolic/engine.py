"""Symbolic engine: siapkan rule-frame, hitung threshold leak-free, terapkan rule.

Threshold kohort (Q1 klik mingguan, median early-engagement) WAJIB di-fit dari train
saja (lihat RuleThresholds.fit) agar tidak bocor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nsoulad.data.schema import UNIT_KEYS
from nsoulad.data.windowing import WindowView
from nsoulad.features.assessment import assessment_features
from nsoulad.features.course_structure import course_structure_features
from nsoulad.features.temporal import weekly_engagement
from nsoulad.symbolic.rules import (
    RULE_IDS,
    RULE_META,
    evaluate_rules,
    severity_counts,
    symbolic_score,
)

_KEYS = list(UNIT_KEYS)
_COURSE = ["code_module", "code_presentation"]


def _sid(df: pd.DataFrame) -> pd.Series:
    """Kunci string per mahasiswa untuk groupby cepat."""
    return (
        df["code_module"].astype(str)
        + "|" + df["code_presentation"].astype(str)
        + "|" + df["id_student"].astype(str)
    )


def _max_consecutive_ones(weekly: pd.DataFrame, flag_col: str) -> pd.Series:
    """Run terpanjang nilai 1 berturut (terurut minggu) per mahasiswa.

    Returns Series indeks = sid unik.
    """
    d = weekly.sort_values(_KEYS + ["week"]).copy()
    d["sid"] = _sid(d)
    # Reset run setiap kali flag==0: cumulative count of zeros sebagai group id.
    d["_grp"] = (d[flag_col] == 0).groupby(d["sid"]).cumsum()
    ones = d[d[flag_col] == 1]
    if len(ones) == 0:
        return pd.Series(dtype=int)
    run = ones.groupby(["sid", "_grp"]).size()
    return run.groupby(level="sid").max()


@dataclass
class RuleThresholds:
    clicks_q1: dict          # (module,presentation) -> Q1 klik mingguan (train)
    early_median: dict       # (module,presentation) -> median total klik (train)
    global_clicks_q1: float
    global_early_median: float

    @classmethod
    def fit(cls, view: WindowView, train_ids, days_per_week: int = 7,
            low_quantile: float = 0.25) -> "RuleThresholds":
        """low_quantile: kuantil klik mingguan untuk R2 (default Q1; diubah hanya untuk
        analisis sensitivitas ambang)."""
        weekly = weekly_engagement(view, days_per_week)
        wk = weekly[weekly["id_student"].isin(set(train_ids))]
        if len(wk) == 0:
            return cls({}, {}, 0.0, 0.0)
        q1 = wk.groupby(_COURSE)["clicks"].quantile(low_quantile)
        tot = wk.groupby(_COURSE + ["id_student"])["clicks"].sum().reset_index()
        med = tot.groupby(_COURSE)["clicks"].median()
        return cls(
            clicks_q1={tuple(k): v for k, v in q1.items()} if isinstance(q1.index, pd.MultiIndex) else q1.to_dict(),
            early_median={tuple(k): v for k, v in med.items()} if isinstance(med.index, pd.MultiIndex) else med.to_dict(),
            global_clicks_q1=float(wk["clicks"].quantile(low_quantile)),
            global_early_median=float(tot["clicks"].median()),
        )

    def q1_for(self, module: str, presentation: str) -> float:
        return float(self.clicks_q1.get((module, presentation), self.global_clicks_q1))

    def median_for(self, module: str, presentation: str) -> float:
        return float(self.early_median.get((module, presentation), self.global_early_median))


@dataclass
class RuleResult:
    keys: pd.DataFrame          # UNIT_KEYS selaras urutan baris
    triggered: pd.DataFrame     # [n,7] boolean(int)
    severity: pd.DataFrame      # rule_count, high/medium_severity_count
    p_symbolic: np.ndarray      # [n]
    rule_frame: pd.DataFrame    # nilai mentah untuk audit/explanation

    def explanations(self) -> list[list[dict]]:
        """Untuk tiap mahasiswa: daftar {rule, reason, action, severity} yang aktif."""
        result = []
        trig = self.triggered[RULE_IDS].to_numpy()
        for row in trig:
            expl = [
                {
                    "rule": rid,
                    "reason": RULE_META[rid].reason,
                    "action": RULE_META[rid].action,
                    "severity": RULE_META[rid].severity,
                }
                for rid, on in zip(RULE_IDS, row) if on
            ]
            result.append(expl)
        return result

    def activation_rate(self) -> pd.Series:
        """Proporsi mahasiswa yang memicu tiap rule (untuk laporan per window)."""
        return self.triggered[RULE_IDS].mean()

    def to_frame(self) -> pd.DataFrame:
        """Gabungkan keys + p_symbolic + severity + triggered untuk join antar-split."""
        df = self.keys.copy()
        df["p_symbolic"] = self.p_symbolic
        df["rule_count"] = self.severity["rule_count"].to_numpy()
        df["high_severity_count"] = self.severity["high_severity_count"].to_numpy()
        for r in RULE_IDS:
            df[r] = self.triggered[r].to_numpy()
        return df


class SymbolicEngine:
    """Terapkan symbolic rules atas sebuah WindowView dengan threshold tertentu."""

    def __init__(self, thresholds: RuleThresholds, weights: dict[str, float] | None = None,
                 days_per_week: int = 7, active_rules: list[str] | None = None,
                 rule_params: dict | None = None):
        self.thresholds = thresholds
        self.weights = weights
        self.days_per_week = days_per_week
        self.active_rules = active_rules
        self.rule_params = rule_params

    def _prepare_rule_frame(self, view: WindowView) -> pd.DataFrame:
        keys = view.cohort_keys[_KEYS].drop_duplicates().reset_index(drop=True)
        weekly = weekly_engagement(view, self.days_per_week).copy()

        # Ambang Q1 per course (merge, fallback global) — hindari apply baris-demi-baris.
        q1_df = pd.DataFrame(
            [(m, p, v) for (m, p), v in self.thresholds.clicks_q1.items()],
            columns=_COURSE + ["q1_course"],
        )
        if len(q1_df):
            weekly = weekly.merge(q1_df, on=_COURSE, how="left")
            weekly["q1"] = weekly["q1_course"].fillna(self.thresholds.global_clicks_q1)
        else:
            weekly["q1"] = self.thresholds.global_clicks_q1
        weekly["below_q1"] = (weekly["clicks"] < weekly["q1"]).astype(int)
        weekly["zero_active"] = (weekly["active_days"] == 0).astype(int)

        run_zero = _max_consecutive_ones(weekly, "zero_active")
        run_below = _max_consecutive_ones(weekly, "below_q1")
        total_clicks = weekly.groupby(_KEYS)["clicks"].sum()

        # Frame per mahasiswa.
        rf = keys.copy()
        sid = _sid(rf)
        rf["run_zero_active"] = sid.map(run_zero).fillna(0).to_numpy()
        rf["run_below_q1"] = sid.map(run_below).fillna(0).to_numpy()
        rf = rf.merge(total_clicks.rename("early_engagement").reset_index(), on=_KEYS, how="left")
        rf["early_engagement"] = rf["early_engagement"].fillna(0.0)

        med_df = pd.DataFrame(
            [(m, p, v) for (m, p), v in self.thresholds.early_median.items()],
            columns=_COURSE + ["median_course"],
        )
        if len(med_df):
            rf = rf.merge(med_df, on=_COURSE, how="left")
            rf["early_median_course"] = rf["median_course"].fillna(self.thresholds.global_early_median)
        else:
            rf["early_median_course"] = self.thresholds.global_early_median

        # Assessment + course-structure + static signals.
        assess = assessment_features(view)[_KEYS + ["first_score", "has_assessment", "n_scored", "n_late"]]
        course = course_structure_features(view, self.days_per_week)[
            _KEYS + ["scheduled_resource_ratio", "active_days_7d_pre_deadline"]
        ]
        static = view.static[_KEYS + ["num_of_prev_attempts"]].copy()
        static["num_of_prev_attempts"] = pd.to_numeric(static["num_of_prev_attempts"], errors="coerce").fillna(0)

        rf = rf.merge(assess, on=_KEYS, how="left").merge(course, on=_KEYS, how="left").merge(static, on=_KEYS, how="left")

        # has_due_assessment: apakah course mahasiswa punya assessment due di window.
        if len(view.assessments_due):
            due_courses = view.assessments_due[_COURSE].drop_duplicates()
            due_courses["has_due_assessment"] = 1
            rf = rf.merge(due_courses, on=_COURSE, how="left")
        else:
            rf["has_due_assessment"] = 0
        rf["has_due_assessment"] = rf["has_due_assessment"].fillna(0).astype(int)

        fill0 = ["first_score", "has_assessment", "n_scored", "n_late", "scheduled_resource_ratio",
                 "active_days_7d_pre_deadline", "num_of_prev_attempts"]
        rf[fill0] = rf[fill0].fillna(0.0)
        return rf

    def apply(self, view: WindowView) -> RuleResult:
        rf = self._prepare_rule_frame(view)
        triggered = evaluate_rules(rf, active_rules=self.active_rules, params=self.rule_params)
        sev = severity_counts(triggered)
        p_sym = symbolic_score(triggered, self.weights)
        return RuleResult(
            keys=view.cohort_keys[_KEYS].drop_duplicates().reset_index(drop=True),
            triggered=triggered,
            severity=sev,
            p_symbolic=p_sym,
            rule_frame=rf,
        )
