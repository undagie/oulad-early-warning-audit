# Pre-Registration — Making the Neuro-Symbolic Contribution Visible

**Locked before running any experiment.** Scope for this round (user-selected):
**Phase 0 diagnosis → Track A (uncertainty-gated fusion) → Track D1 (calibration).**
Integrity stance: hypotheses, metrics, strata definitions, and decision rules are fixed
*here* before results are seen; **every** outcome — including nulls — is reported.

---

## 0. Problem statement & diagnosis (why NS-GRU = GRU on full data)

Parity is mechanistically expected, for three reasons visible in the code:

1. **Information redundancy.** The GRU already ingests every raw signal the rules are
   computed from (weekly clickstream → R1/R2/R7; tabular assessment/course → R3–R6, via
   `_SeqNet.head`, `models/gru.py`). `P_symbolic` is therefore a coarse, thresholded
   restatement of what the neural encoder already extracts → collinear → discounted.
2. **Averaging kills a conditional benefit.** `LogisticFusion` (`models/fusion.py`) applies
   **one global coefficient** to ~20k students; it cannot say "trust the rules *only here*."
3. **No gradient path (late fusion).** The GRU never sees the rules during training, so it
   cannot specialise to complement them.

The improvement strategy attacks reasons #2 (Track A) and adds a new evaluation axis (D1),
without weakening the existing honest negative result.

---

## 1. Pre-registered hypotheses

**H-A1 — Conditional complementarity (existence).**
There is an *a priori identifiable* subgroup where symbolic evidence carries predictive
signal **not already captured** by the neural score. Operationalised as: within the
**neural-uncertain band** (P_neural ∈ [Q0.30, Q0.70] of the cohort's P_neural), adding
{P_symbolic, rule_count} to P_neural raises ROC-AUC for the at-risk label.
- Direction: ΔAUC_residual > 0. Confirmed if ΔAUC > 0 with paired-bootstrap p < 0.05 in
  ≥ 2 of 4 windows (mean over seeds).

**H-A2 — Exploitable gain (gated fusion).**
An uncertainty-gated fusion converts that latent signal into measured accuracy:
- **Primary endpoint:** subgroup ΔAUC (gated-NS-GRU − GRU) on the neural-uncertain band,
  weeks 2 & 8, 5 seeds, paired bootstrap, Holm-corrected. Success: ΔAUC > 0, Holm p < 0.05.
- **Guardrail (must also hold):** full-cohort non-inferiority vs GRU, margin **δ = 0.005**
  (full-cohort AUC must not drop by more than δ).

**H-D1 — Calibration.**
The neuro-symbolic model is better *calibrated* than the pure neural / boosted baseline,
even where AUC is tied.
- **Primary metric:** ECE (15 equal-frequency bins), lower is better.
- Success: NS-GRU (or gated variant) has lower ECE than GRU in ≥ 3 of 4 windows
  (mean over seeds), with Brier as confirmatory secondary.

## 2. Metrics & endpoints

| Endpoint | Metric | Test |
|---|---|---|
| Primary (A) | subgroup ROC-AUC, gated-NS-GRU vs GRU, uncertain band | paired bootstrap B=2000, Holm |
| Guardrail (A) | full-cohort ROC-AUC non-inferiority (δ=0.005) vs GRU | paired bootstrap |
| Secondary (A) | risk–coverage AURC (selective prediction) | descriptive + bootstrap |
| Primary (D1) | ECE (15-bin equal-freq) | mean±std over seeds |
| Secondary (D1) | Brier, MCE, reliability-curve slope | descriptive |
| Diagnostic (Phase 0) | ΔAUC_residual in uncertain band; per-decile complementarity map | bootstrap |

## 3. Locked protocol parameters

- **Seeds:** 42–46 (5) for headline endpoints; 42–44 (3) allowed for exploratory Phase 0.
- **Windows:** 2, 4, 6, 8. **Label/split/LEAP protocol:** unchanged from the paper.
- **Decision threshold:** tuned on validation for balanced accuracy (unchanged; never 0.5).
- **Neural-uncertain band:** P_neural ∈ [Q0.30, Q0.70], computed **per (seed, window)** on the
  test cohort's P_neural distribution (adaptive, no magic constant).
- **Low-data stratum (when features available):** bottom tercile of total observed clicks in
  the window. (Phase 0 on cached predictions uses the uncertain band + `rule_count` as proxy.)
- **Significance:** paired bootstrap (B=2000) for AUC diffs; McNemar for classification;
  **Holm–Bonferroni** across the comparisons listed in §1–2.
- **Calibration:** 15 equal-frequency bins; ECE/Brier/MCE.

## 4. Track A designs to be evaluated (both auditable)

- **A1 — Auditable interaction fusion.** Extend `LogisticFusion` feature set with
  interaction terms: `P_sym × u`, `rule_count × u`, where `u = neural uncertainty`
  (e.g., `1 − |2·P_neural − 1|`). Coefficients remain inspectable — the paper's audit story
  is preserved.
- **A2 — Confidence router / mixture.** `g = σ(w·[u, n_weeks, log(1+clicks)])`;
  `P_final = g · P_sym* + (1 − g) · P_neural`, with `P_sym*` isotonic-calibrated. Report the
  learned gate as a function of uncertainty/data-sufficiency.

## 5. Decision rules (fixed in advance)

- **Full success:** H-A1 confirmed **and** H-A2 primary significant (Holm) **and** guardrail
  non-inferiority holds → gated-NS-GRU becomes a headline result; reframe contribution as
  "diagnosed redundancy, then recovered a conditional gain," keeping the full-data-average
  negative result intact and explained.
- **Partial:** H-A1 confirmed but H-A2 not significant → report "latent but not exploitable by
  a linear/gated fuser"; motivates Track B (consistency-loss GRU) as future work.
- **Null:** H-A1 rejected (no residual signal in any window) → **abandon Track A**, do not
  chase full-data AUC; rest the contribution on Track D (calibration + robustness) and keep
  the existing honest negative accuracy result unchanged.
- **Calibration (D1):** reported regardless of A's outcome; a positive ECE result is a
  standalone contribution even if AUC stays tied.

## 6. What we will report no matter what

All windows, all seeds, all three decision branches, and any null. No post-hoc threshold or
band re-tuning after seeing outcomes. Any deviation from this document is logged with a reason
in `outputs/diagnosis_log.txt`.
