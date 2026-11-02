"""Definisi 7 symbolic pedagogical rules (R1..R7) dan evaluasinya.

Tiap rule mengevaluasi satu baris-per-mahasiswa frame fitur (`rf`) dan menghasilkan
boolean triggered. Metadata (bobot, severity, aksi) mengikuti desain eksperimen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Ambang domain (eksplisit, dirujuk dari desain).
LOW_SCORE_THRESHOLD = 40.0
SCHEDULED_RATIO_THRESHOLD = 0.30
CONSECUTIVE_WEEKS = 2


@dataclass(frozen=True)
class RuleSpec:
    rid: str
    name: str
    weight: float
    severity: str  # "low" | "medium" | "high"
    reason: str    # kalimat alasan untuk explanation
    action: str    # saran intervensi


RULE_META: dict[str, RuleSpec] = {
    "R1": RuleSpec("R1", "Persistent Inactivity", 0.25, "high",
                   "No VLE activity for two consecutive weeks.",
                   "Tutor sends a check-in message."),
    "R2": RuleSpec("R2", "Low Engagement", 0.15, "medium",
                   "Weekly clicks below peers' Q1 for two consecutive weeks.",
                   "Recommend a weekly learning plan."),
    "R3": RuleSpec("R3", "Missing Scheduled Resources", 0.15, "medium",
                   "Accessed less than 30% of scheduled resources.",
                   "Recommend specific learning materials."),
    "R4": RuleSpec("R4", "Poor Early Assessment", 0.30, "high",
                   "First assessment score below 40.",
                   "Offer remedial support."),
    "R5": RuleSpec("R5", "Late Assessment Submission", 0.15, "medium",
                   "Assessment submitted after the deadline.",
                   "Time-management intervention."),
    "R6": RuleSpec("R6", "Repeated Attempt With Low Engagement", 0.10, "medium",
                   "Repeated module attempt with below-median early engagement.",
                   "Tutor reviews prior difficulty."),
    "R7": RuleSpec("R7", "No Activity Before Assessment", 0.20, "high",
                   "No VLE activity in the 7 days before each assessment deadline so far.",
                   "Plan recovery before the next deadline."),
}

RULE_IDS = list(RULE_META)


def max_possible_weight() -> float:
    return float(sum(spec.weight for spec in RULE_META.values()))


# Rule yang memanfaatkan struktur course (scheduled resources & assessment deadline).
COURSE_STRUCTURE_RULES = ["R3", "R7"]


def evaluate_rules(rf: pd.DataFrame, active_rules: list[str] | None = None,
                   params: dict | None = None) -> pd.DataFrame:
    """Evaluasi R1..R7 atas frame per-mahasiswa `rf`. Returns DataFrame [n,7] boolean.

    `rf` wajib memuat kolom: run_zero_active, run_below_q1, scheduled_resource_ratio,
    first_score, has_assessment, n_late, num_of_prev_attempts, early_engagement,
    early_median_course, has_due_assessment, active_days_7d_pre_deadline.

    active_rules: jika diberikan, rule di luar daftar di-nonaktifkan (di-nol-kan).
    Dipakai untuk ablation (mis. NS-GRU tanpa course-structure rules).

    params: override ambang untuk analisis sensitivitas, kunci opsional
    consecutive_weeks, scheduled_ratio, low_score, min_late (R5 memicu bila
    n_late >= min_late; default 1 = definisi asli).
    """
    p = params or {}
    weeks = p.get("consecutive_weeks", CONSECUTIVE_WEEKS)
    out = pd.DataFrame(index=rf.index)
    out["R1"] = rf["run_zero_active"] >= weeks
    out["R2"] = rf["run_below_q1"] >= weeks
    out["R3"] = rf["scheduled_resource_ratio"] < p.get("scheduled_ratio", SCHEDULED_RATIO_THRESHOLD)
    # R4 butuh skor yang sudah terlihat; submission tanpa skor (NaN/tertahan lag) tidak memicu.
    has_score = rf["n_scored"] > 0 if "n_scored" in rf else rf["has_assessment"] == 1
    out["R4"] = has_score & (rf["first_score"] < p.get("low_score", LOW_SCORE_THRESHOLD))
    out["R5"] = rf["n_late"] >= p.get("min_late", 1)
    out["R6"] = (rf["num_of_prev_attempts"] > 0) & (rf["early_engagement"] < rf["early_median_course"])
    out["R7"] = (rf["has_due_assessment"] == 1) & (rf["active_days_7d_pre_deadline"] == 0)
    out = out.astype(int)
    if active_rules is not None:
        for r in RULE_IDS:
            if r not in active_rules:
                out[r] = 0
    return out


def severity_counts(triggered: pd.DataFrame) -> pd.DataFrame:
    """Hitung jumlah rule per severity yang aktif (fitur untuk fusion)."""
    high = [r for r, s in RULE_META.items() if s.severity == "high"]
    med = [r for r, s in RULE_META.items() if s.severity == "medium"]
    return pd.DataFrame(
        {
            "rule_count": triggered[RULE_IDS].sum(axis=1).to_numpy(),
            "high_severity_count": triggered[high].sum(axis=1).to_numpy(),
            "medium_severity_count": triggered[med].sum(axis=1).to_numpy(),
        },
        index=triggered.index,
    )


def symbolic_score(triggered: pd.DataFrame, weights: dict[str, float] | None = None) -> np.ndarray:
    """P_symbolic = weighted_sum(triggered) / max_possible_weight."""
    w = weights or {r: RULE_META[r].weight for r in RULE_IDS}
    wmax = sum(w.values())
    weighted = sum(triggered[r].to_numpy() * w[r] for r in RULE_IDS)
    return weighted / wmax if wmax > 0 else np.zeros(len(triggered))
