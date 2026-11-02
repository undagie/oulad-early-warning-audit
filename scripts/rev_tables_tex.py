"""Revisi SOIC 4706: tulis tabel LaTeX naskah dari outputs/revision/tables/*.csv.

Setiap tabel ditulis ke submission_soic/revision/tables/<nama>.tex dan di-\\input dari naskah,
sehingga angka di naskah selalu sama dengan hasil analisis. Tabel yang CSV-nya belum ada dilewati.

    python scripts/rev_tables_tex.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "revision" / "tables"
OUT = ROOT / "submission_soic" / "revision" / "tables"
WEEKS = [2, 4, 6, 8]
NAME = {"NS-GRU-2in": "NS-GRU", "NS-GRU": "NS-GRU-4in"}
# rule names exactly as in the manuscript's rule table (tab:rules)
RULE = {"R1": "R1 Persistent inactivity", "R2": "R2 Low engagement", "R3": "R3 Missing resources",
        "R4": "R4 Poor early score", "R5": "R5 Late submission", "R6": "R6 Repeat + low engagement",
        "R7": "R7 Pre-deadline inactivity"}


def _read(name: str) -> pd.DataFrame | None:
    p = TAB / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _num(x: float, d: int = 3, sign: bool = False) -> str:
    if pd.isna(x):
        return "--"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")


def _pm(m: float, s: float, d: int = 3) -> str:
    return f"${m:.{d}f} \\pm {s:.{d}f}$"


def _ci(lo: float, hi: float, d: int = 4) -> str:
    return f"[${lo:+.{d}f}$, ${hi:+.{d}f}$]"


def _d4(x: float) -> str:
    """Signed 4-decimal value; a value that rounds to zero is printed without a sign."""
    return "$0.0000$" if abs(x) < 5e-5 else f"${x:+.4f}$"


def _int(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", "{,}")


def _write(name: str, caption: str, label: str, body: str, size: str = "\\small",
           note: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tex = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{\\rev{{{caption}}}}}\\label{{{label}}}",
           size, "\\color{blue}", body.strip()]
    if note:
        tex += ["\\par\\smallskip", f"\\parbox{{0.94\\textwidth}}{{\\footnotesize {note}}}"]
    tex.append("\\end{table}")
    (OUT / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  -> tables/{name}.tex")


# ---------------------------------------------------------------------------------------------- cohort
def cohort() -> None:
    c = _read("cohort_accounting")
    if c is None:
        return
    rows = [f"{int(r.window)} & {int(r.cutoff_day)} & {_int(r.registered_by_cutoff)} & "
            f"{_int(r.excluded_withdrawn_before_cutoff)} & {r.excluded_share_of_all_withdrawn * 100:.1f} & "
            f"{_int(r.scored_cohort)} & {r.scored_base_rate:.3f} \\\\" for r in c.itertuples()]
    f = c.iloc[0]
    body = ("\\begin{tabular}{rrrrrrr}\n\\toprule\n"
            "Week & Cutoff day & Registered & Excluded & \\% of Withdrawn & Scored & Base rate \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\midrule\n"
            f"\\multicolumn{{5}}{{l}}{{Fixed cohort (registered by week 2, still registered at week 8)}} & "
            f"{_int(f.fixed_cohort)} & {f.fixed_base_rate:.3f} \\\\\n\\bottomrule\n\\end{{tabular}}")
    _write("tab_cohort",
           "Cohort accounting under the primary cohort rule. Registered counts enrolments registered by the "
           "cutoff; Excluded counts those that had already unregistered by the cutoff; \\% of Withdrawn is the "
           "share of all 10{,}156 Withdrawn outcomes in OULAD that the excluded enrolments carry; Scored is the "
           "cohort that receives a prediction.",
           "tab:cohort", body)


# ---------------------------------------------------------------------------------------------- predictive
def predictive() -> None:
    p = _read("main_predictive")
    if p is None:
        return
    order = ["NS-GRU-2in", "NS-GRU", "NS-GRU-RA", "GRU", "LSTM", "Transformer", "XGBoost", "Rules-only",
             "Static-only"]
    lines = []
    for metric, panel in (("auc", "Panel A: ROC-AUC"), ("pr_auc", "Panel B: PR-AUC")):
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{panel}}}}} \\\\")
        for m in order:
            g = p[p.model == m].set_index("window").reindex(WEEKS)
            cells = " & ".join(_pm(g.loc[w, f"{metric}_mean"], g.loc[w, f"{metric}_std"]) for w in WEEKS)
            lines.append(f"{NAME.get(m, m)} & {cells} \\\\")
        if metric == "auc":
            lines.append("\\midrule")
    body = ("\\renewcommand{\\arraystretch}{0.94}\n\\begin{tabular}{lrrrr}\n\\toprule\n"
            "Model & Week 2 & Week 4 & Week 6 & Week 8 \\\\\n\\midrule\n" + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}")
    _write("tab_predictive",
           "Test discrimination over five student-level splits (mean $\\pm$ standard deviation), with "
           "per-seed model selection. NS-GRU is the standardized two-input fusion; NS-GRU-4in is the "
           "four-input fusion of the submitted version.", "tab:predictive", body, size="\\footnotesize")


def aucdiff() -> None:
    d = _read("main_paired_auc")
    if d is None:
        return
    comps = [("NS-GRU-2in - GRU", "NS-GRU $-$ GRU"), ("NS-GRU-2in - XGBoost", "NS-GRU $-$ XGBoost"),
             ("NS-GRU-2in - Static-only", "NS-GRU $-$ Static-only")]
    lines = []
    for key, lab in comps:
        g = d[d.comparison == key].set_index("window").reindex(WEEKS)
        for w in WEEKS:
            r = g.loc[w]
            ptost = "$<0.001$" if r.p_tost < 0.001 else f"{r.p_tost:.3f}"
            eq = "yes" if r.equivalent else "no"
            lines.append(f"{lab if w == 2 else ''} & {w} & {_d4(r['mean'])} & {_ci(r.ci95_low, r.ci95_high)} & "
                         f"{_ci(r.ci90_low, r.ci90_high)} & {ptost} & {eq} \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]
    body = ("\\begin{tabular}{llrccrc}\n\\toprule\n"
            "Comparison & Week & Mean & 95\\% CI & 90\\% CI & $p_{\\mathrm{TOST}}$ & Equivalent \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    _write("tab_aucdiff",
           "Paired test ROC-AUC differences over five splits. Intervals use the corrected resampled $t$ "
           "statistic ($J=5$, 4 degrees of freedom); $p_{\\mathrm{TOST}}$ is the larger of the two one-sided "
           "$p$-values for the equivalence margin $\\pm0.01$, and equivalence is declared when the 90\\% "
           "interval lies inside the margin.", "tab:aucdiff", body, size="\\footnotesize")


def sensitivity() -> None:
    s = _read("sensitivity_vs_main")
    if s is None:
        return

    def cell(cond: str, model: str, w: int, d: int) -> str:
        r = s[(s.condition == cond) & (s.model == model) & (s.window == w)]
        if not len(r):
            return "--"
        r = r.iloc[0]
        if abs(r.delta_mean) < 5e-7 and abs(r.delta_ci_low) < 5e-7 and abs(r.delta_ci_high) < 5e-7:
            return "$0$ (identical)"
        if pd.isna(r.delta_ci_low):
            return f"${r.delta_mean:+.{d}f}$"
        # Interval on a second line so that the four week columns fit the text width
        return (f"\\begin{{tabular}}{{@{{}}c@{{}}}}${r.delta_mean:+.{d}f}$\\\\ "
                f"{{[}}${r.delta_ci_low:+.{d}f}$, ${r.delta_ci_high:+.{d}f}${{]}}\\end{{tabular}}")

    def absval(cond: str, model: str, w: int) -> str:
        r = s[(s.condition == cond) & (s.model == model) & (s.window == w)]
        return f"{r.auc_mean.iloc[0]:.3f}" if len(r) else "--"

    lines = ["\\multicolumn{5}{l}{\\textit{Panel A: admitting post-cutoff evidence (leaky $-$ strict)}} \\\\"]
    for m in ("NS-GRU-2in", "XGBoost", "Rules-only", "Static-only"):
        lines.append(f"{NAME.get(m, m)} & " + " & ".join(cell("leaky", m, w, 3) for w in WEEKS) + " \\\\")
    lines.append("NS-GRU, leaky AUC & " + " & ".join(absval("leaky", "NS-GRU-2in", w) for w in WEEKS) + " \\\\")
    lines += ["\\midrule", "\\multicolumn{5}{l}{\\textit{Panel B: delayed score visibility, NS-GRU (lag $-$ primary)}} \\\\"]
    for lag in ("lag7", "lag14", "lag21"):
        lines.append(f"{lag[3:]}-day lag & " + " & ".join(cell(lag, "NS-GRU-2in", w, 4) for w in WEEKS) + " \\\\")
    lines += ["\\midrule", "\\multicolumn{5}{l}{\\textit{Panel C: cohort definition, NS-GRU (alternative $-$ primary)}} \\\\"]
    for cond, lab in (("enrolment_cohort", "Enrolment-defined cohort"), ("fixed_cohort", "Fixed cohort")):
        lines.append(f"{lab} & " + " & ".join(cell(cond, "NS-GRU-2in", w, 3) for w in WEEKS) + " \\\\")
        lines.append(f"\\quad AUC & " + " & ".join(absval(cond, "NS-GRU-2in", w) for w in WEEKS) + " \\\\")
    body = ("\\begin{tabular}{lcccc}\n\\toprule\n & Week 2 & Week 4 & Week 6 & Week 8 \\\\\n\\midrule\n" +
            "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    _write("tab_sensitivity",
           "Temporal-validity and cohort sensitivity of test ROC-AUC over five seeds: mean difference from the "
           "primary analysis with a corrected resampled 95\\% interval. The leaky condition admits post-cutoff "
           "assessment and course-structure records while keeping the cohort and the $k$-week engagement "
           "sequence fixed; Static-only is the temporally invariant control.", "tab:sensitivity", body,
           size="\\scriptsize")


# ---------------------------------------------------------------------------------------------- reliance
def fusioncoef() -> None:
    import json

    rows = []
    for f in sorted((ROOT / "outputs" / "revision" / "preds" / "main").glob("*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        c = m["fusion_coef"]["compact"]
        rows.append({"window": m["window"], "seed": m["seed"], "g1": c["P_neural"], "g2": c["P_symbolic"]})
    if not rows:
        return
    r = pd.DataFrame(rows)
    r["ratio"] = r.g2.abs() / r.g1.abs()
    ra = _read("rank_agreement")
    fo = _read("flag_overlap")
    pa = _read("main_paired_auc")
    lines = []
    for w in WEEKS:
        g = r[r.window == w]
        cells = [_pm(g.g1.mean(), g.g1.std(), 2), _pm(g.g2.mean(), g.g2.std(), 2),
                 _pm(g.ratio.mean(), g.ratio.std(), 2)]
        if ra is not None:
            k = ra[(ra.window == w) & (ra.pair == "NS-GRU-2in vs GRU")]
            cells.append(f"{k.spearman.min():.3f}")
        if fo is not None:
            k = fo[(fo.window == w) & (fo.pair == "NS-GRU-2in vs GRU") & (fo.capacity == 0.10)]
            cells.append(f"{k.overlap.mean():.3f}")
        if pa is not None:
            k = pa[(pa.window == w) & (pa.comparison == "NS-GRU-2in - GRU")].iloc[0]
            cells.append(_d4(k["mean"]))
        lines.append(f"{w} & " + " & ".join(cells) + " \\\\")
    head = ["Week", "$\\gamma_1$ ($\\Pn$)", "$\\gamma_2$ ($\\Ps$)", "$|\\gamma_2|/|\\gamma_1|$"]
    if ra is not None:
        head.append("Min. Spearman")
    if fo is not None:
        head.append("Top-10\\% overlap")
    if pa is not None:
        head.append("$\\Delta$AUC")
    body = (f"\\begin{{tabular}}{{l{'r' * (len(head) - 1)}}}\n\\toprule\n" + " & ".join(head) +
            " \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    _write("tab_fusioncoef",
           "Branch reliance of NS-GRU over five seeds. Coefficients are per standard deviation of each input "
           "(mean $\\pm$ standard deviation); Min. Spearman is the lowest rank correlation between the NS-GRU "
           "and GRU test scores over the five splits; Top-10\\% overlap is the mean share of the NS-GRU top decile "
           "that GRU also flags; $\\Delta$AUC is NS-GRU minus GRU.", "tab:fusioncoef", body)


def band() -> None:
    b = _read("band_diagnostic")
    if b is None:
        return
    cols = b.columns
    lines = []
    for r in b.itertuples():
        lines.append(f"{int(r.window)} & {_int(r.n_band)} & {r.auc_neural:.3f} & {r.auc_symbolic:.3f} & "
                     f"${r.delta_auc:+.4f}$ & {_ci(r.delta_ci_low, r.delta_ci_high)} \\\\")
    body = ("\\begin{tabular}{rrrrrc}\n\\toprule\n"
            "Week & $n$ (band) & AUC $\\Pn$ & AUC $\\Ps$ alone & $\\Delta$AUC, rules added & 95\\% CI \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    _write("tab_band",
           "Residual symbolic signal in the neural-uncertain band $[Q_{0.30}, Q_{0.70}]$ of the GRU score in "
           "each test split, pooled over five splits. $\Delta$AUC compares an in-sample logistic fit of "
           "$\Pn$, $\Ps$, and the two rule counts with $\Pn$ alone and is descriptive; intervals come from a "
           "bootstrap that resamples students ($B=500$).", "tab:diagnosis", body)
    _ = cols


# ---------------------------------------------------------------------------------------------- alerts
def alerts() -> None:
    lo = _read("rule_leave_one_out")
    dg = _read("disagreement")
    if lo is None or dg is None:
        return
    a = lo[lo.rule_set == "all"]
    lines = []
    for w in WEEKS:
        x, y = a[a.window == w], dg[dg.window == w]
        lines.append(f"{w} & {_pm(x.coverage.mean(), x.coverage.std())} & {_pm(x.sparsity.mean(), x.sparsity.std(), 2)}"
                     f" & {_pm(y.kendall_tau.mean(), y.kendall_tau.std())} & "
                     f"{_pm(y.alert_low_risk.mean(), y.alert_low_risk.std())} & "
                     f"{_pm(y.silent_high_risk.mean(), y.silent_high_risk.std())} \\\\")
    body = ("\\begin{tabular}{lrrrrr}\n\\toprule\n"
            "Week & Coverage & Sparsity & Kendall $\\tau$ & Alert, low risk & Silent, high risk \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    _write("tab_alerts",
           "Availability and disagreement of the rule-based alerts over five seeds (mean $\\pm$ standard "
           "deviation). Coverage is the share of NS-GRU positives at the validation threshold with at least one "
           "alert; sparsity is the mean number of alerts per covered positive; Kendall $\\tau$ is computed between "
           "$\\Ps$ and the recalibrated neural risk; Alert, low risk is the share of enrolments with a "
           "high-severity alert whose recalibrated risk is below the base rate; Silent, high risk is the share "
           "of the highest-risk decile with no alert.", "tab:alerts", body, size="\\footnotesize")


def diagnostic() -> None:
    rm = _read("rule_metrics")
    if rm is None:
        return
    w8 = rm[rm.window == 8].sort_values("precision", ascending=False)
    w2 = rm[rm.window == 2].set_index("rule")
    lines = []
    for r in w8.itertuples():
        a2 = w2.activation.get(r.rule, np.nan)
        lines.append(f"{RULE[r.rule]} & {a2:.3f} & "
                     f"{r.activation:.3f} & {r.precision:.3f} [{r.precision_ci_low:.3f}, {r.precision_ci_high:.3f}] & "
                     f"{r.recall:.3f} & {r.lift:.2f} \\\\")
    br = w8.base_rate.iloc[0]
    body = ("\\begin{tabular}{lrrcrr}\n\\toprule\n"
            "Rule & Act. W2 & Act. W8 & Precision W8 [95\\% CI] & Recall W8 & Lift W8 \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    _write("tab_diagnostic",
           f"Diagnostic validity of each rule-based alert on the pooled test splits (base rate at week 8: "
           f"{br:.3f}). Act. is the activation rate; precision is $P(\\text{{at risk}}\\mid\\text{{alert}})$ with a "
           "student-clustered bootstrap interval; recall is $P(\\text{alert}\\mid\\text{at risk})$; lift is "
           "precision divided by the base rate.", "tab:diagnostic", body, size="\\footnotesize")


def xai() -> None:
    ga = _read("explainer_grouped_attribution")
    ds = _read("explainer_dice_summary")
    if ga is None or ds is None:
        return
    sh, ts = "Grouped Shapley (fused NS-GRU)", "Grouped TreeSHAP (XGBoost)"

    def att(expl: str, group: str, col: str) -> str:
        r = ga[(ga.explainer == expl) & (ga.group == group)].set_index("window").reindex(WEEKS)
        return " & ".join(f"{v * 100:.1f}" for v in r[col])

    def dice(mode: str, col: str, fmt: str) -> str:
        r = ds[ds["mode"] == mode].set_index("window").reindex(WEEKS)
        return " & ".join(fmt.format(v) for v in r[col])

    succ = " & ".join(f"{int(r.succeeded)}/{int(r.attempted)}"
                      for r in ds[ds["mode"] == "unconstrained"].set_index("window").reindex(WEEKS).itertuples())
    lines = [
        "\\multicolumn{5}{l}{\\textit{Grouped Shapley, fused NS-GRU (300 flagged enrolments per week)}} \\\\",
        f"Personal attributes, share of attribution (\\%) & {att(sh, 'personal_attributes', 'mean_abs_share')} \\\\",
        f"Personal attributes ranked first (\\% of cases) & {att(sh, 'personal_attributes', 'top1_rate')} \\\\",
        f"Course context, share of attribution (\\%) & {att(sh, 'course_context', 'mean_abs_share')} \\\\",
        f"Rule summary, share of attribution (\\%) & {att(sh, 'rule_summary', 'mean_abs_share')} \\\\",
        "\\multicolumn{5}{l}{\\textit{Grouped TreeSHAP, XGBoost (same enrolments)}} \\\\",
        f"Personal attributes, share of attribution (\\%) & {att(ts, 'personal_attributes', 'mean_abs_share')} \\\\",
        f"Personal attributes ranked first (\\% of cases) & {att(ts, 'personal_attributes', 'top1_rate')} \\\\",
        "\\multicolumn{5}{l}{\\textit{DiCE counterfactuals (100 flagged at-risk enrolments per week)}} \\\\",
        f"Successful generations, either mode & {succ} \\\\",
        f"Unconstrained: change an immutable attribute (\\%) & {dice('unconstrained', 'immutable_change_rate', '{:.0%}').replace('%', '')} \\\\",
        f"Unconstrained: features changed (mean) & {dice('unconstrained', 'mean_features_changed', '{:.2f}')} \\\\",
        f"Mutable-only: features changed (mean) & {dice('mutable-only', 'mean_features_changed', '{:.2f}')} \\\\",
    ]
    body = ("\\begin{tabular}{lrrrr}\n\\toprule\n & Week 2 & Week 4 & Week 6 & Week 8 \\\\\n\\midrule\n" +
            "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    _write("tab_xai",
           "Structural contrast between post-hoc outputs (seed 42). Attribution shares are mean absolute grouped "
           "Shapley values normalized per enrolment; personal attributes are gender, region, age band, IMD band, "
           "disability, prior education, prior attempts, and studied credits; course context is module and "
           "presentation. DiCE runs on an interpretable random-forest model; immutable attributes are the personal "
           "attributes other than studied credits, and mutable-only counterfactuals change none of them by "
           "construction.",
           "tab:xai", body, size="\\footnotesize")


# ---------------------------------------------------------------------------------------------- utility
def earlywarning() -> None:
    op = _read("operating_points")
    if op is None:
        return
    g = op.groupby(["window", "model", "capacity"])[["precision", "recall"]].mean()
    lines = []
    for c in (0.05, 0.10, 0.20, 0.30):
        cells = []
        for w, m in ((2, "NS-GRU-2in"), (8, "NS-GRU-2in"), (8, "XGBoost"), (8, "Static-only")):
            r = g.loc[(w, m, c)]
            cells += [f"{r.precision:.3f}", f"{r.recall:.3f}"]
        lines.append(f"{int(c * 100)}\\% & " + " & ".join(cells) + " \\\\")
    body = ("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
            " & \\multicolumn{2}{c}{NS-GRU, week 2} & \\multicolumn{2}{c}{NS-GRU, week 8} & "
            "\\multicolumn{2}{c}{XGBoost, week 8} & \\multicolumn{2}{c}{Static-only, week 8} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
            "Capacity & Prec. & Rec. & Prec. & Rec. & Prec. & Rec. & Prec. & Rec. \\\\\n\\midrule\n" +
            "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    _write("tab_earlywarning",
           "Precision and recall when a fixed share of enrolments is flagged (mean over five seeds). GRU "
           "differs from NS-GRU by at most 0.004 in every cell.", "tab:earlywarning", body, size="\\footnotesize")


def calibration() -> None:
    c = _read("calibration_all_models")
    if c is None:
        return
    g = c.groupby(["window", "model"])[["ece_raw", "ece_cal", "brier_raw", "brier_cal"]].mean()
    lines = []
    for m in ["NS-GRU-2in", "GRU", "XGBoost", "Static-only", "Rules-only"]:
        cells = []
        for w in (2, 8):
            r = g.loc[(w, m)]
            cells += [f"{r.ece_raw:.3f}", f"{r.ece_cal:.3f}", f"{r.brier_raw:.3f}", f"{r.brier_cal:.3f}"]
        lines.append(f"{NAME.get(m, m)} & " + " & ".join(cells) + " \\\\")
    body = ("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
            " & \\multicolumn{4}{c}{Week 2} & \\multicolumn{4}{c}{Week 8} \\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
            "Model & ECE & ECE$_{\\mathrm{iso}}$ & Brier & Brier$_{\\mathrm{iso}}$ & ECE & ECE$_{\\mathrm{iso}}$ & "
            "Brier & Brier$_{\\mathrm{iso}}$ \\\\\n\\midrule\n" + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}")
    _write("tab_calibration",
           "Calibration before and after isotonic recalibration fitted on the validation split, applied "
           "identically to every model (mean over five seeds; ECE with 15 equal-frequency bins).",
           "tab:calibration", body, size="\\footnotesize")


# ---------------------------------------------------------------------------------------------- fairness
def fairness() -> None:
    f = _read("fairness")
    if f is None:
        return
    w8 = f[(f.window == 8) & (f.attribute != "region")]
    lines, last = [], None
    label = {"gender": "Gender", "disability": "Disability", "age_band": "Age band", "imd_band": "IMD quintile", "imd_quintile": "IMD quintile",
             "highest_education": "Prior education", "prev_attempt": "Prior attempt", "prior_attempt": "Prior attempt",
             "num_of_prev_attempts": "Prior attempt"}
    for r in w8.itertuples():
        att = label.get(r.attribute, str(r.attribute).replace("_", " ")) if r.attribute != last else ""
        if r.attribute != last and last is not None:
            lines.append("\\addlinespace[2pt]")
        last = r.attribute
        grp = str(r.group).replace("%", "\\%").replace("_", " ")
        lines.append(f"{att} & {grp} & {_int(r.n)} & {r.base_rate:.3f} & {r.auc:.3f} [{r.auc_ci_low:.3f}, "
                     f"{r.auc_ci_high:.3f}] & {r.ece:.3f} & {r.fpr_top10:.3f} & {r.fnr_top10:.3f} \\\\")
    body = ("\\begin{tabular}{llrrcrrr}\n\\toprule\n"
            "Attribute & Group & $n$ & Base rate & AUC [95\\% CI] & ECE$_{\\mathrm{iso}}$ & FPR@10\\% & FNR@10\\% \\\\\n"
            "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}")
    _write("tab_fairness",
           "Subgroup audit of NS-GRU at week 8, pooled over the five test splits. AUC intervals resample "
           "students; ECE is computed after isotonic recalibration; FPR and FNR are evaluated at the "
           "top-10\\% operating point of each split. Region (13 groups) is reported in the released results.",
           "tab:fairness", body, size="\\scriptsize")


# ---------------------------------------------------------------------------------------------- generalization
def generalization() -> None:
    t = _read("transfer_disjoint_predictive")
    p = _read("main_predictive")
    lm = _read("lomo_per_seed")
    if t is None or p is None:
        return
    lines = []
    for w in WEEKS:
        cells = []
        for m in ("NS-GRU-2in", "GRU", "XGBoost", "Static-only"):
            r = t[(t.window == w) & (t.model == m)]
            cells.append(_pm(r.auc_mean.iloc[0], r.auc_std.iloc[0]) if len(r) else "--")
        ref = p[(p.window == w) & (p.model == "NS-GRU-2in")].iloc[0]
        cells.append(f"{ref.auc_mean:.3f}")
        if lm is not None:
            x = lm[(lm.window == w) & (lm.model == "NS-GRU-2in")].groupby("held_out_module").auc.mean()
            cells.append(f"{x.mean():.3f} ({x.min():.3f}--{x.max():.3f})" if len(x) else "--")
        lines.append(f"{w} & " + " & ".join(cells) + " \\\\")
    head = "Week & NS-GRU & GRU & XGBoost & Static-only & Student split"
    cols = "lrrrrr"
    if lm is not None:
        head += " & LOMO NS-GRU (range)"
        cols += "c"
    body = (f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head} \\\\\n\\midrule\n" + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}")
    _write("tab_generalization",
           "Identity-disjoint transfer from 2013 to 2014 presentations (test ROC-AUC, mean $\\pm$ standard "
           "deviation over five seeds; every student with a 2013 enrolment is removed from the test set), the "
           "student-split NS-GRU reference, and leave-one-module-out NS-GRU AUC (mean over seven held-out "
           "modules and three seeds, with the range of module means).", "tab:generalization", body,
           size="\\scriptsize")


def dataeff() -> None:
    d = _read("data_efficiency_matched")
    if d is None:
        return
    models = ["NS-GRU-2in", "GRU", "NS-GRU-RA", "XGBoost", "Rules-only"]
    g = d.groupby(["frac", "model"]).auc.agg(["mean", "std"])
    n = d.groupby("frac")[["n_train", "n_val"]].mean()
    lines = []
    for f in sorted(d.frac.unique()):
        cells = [_pm(g.loc[(f, m), "mean"], g.loc[(f, m), "std"]) for m in models]
        lines.append(f"{f * 100:g}\\% & {_int(n.loc[f, 'n_train'])} & {_int(n.loc[f, 'n_val'])} & " +
                     " & ".join(cells) + " \\\\")
    head = " & ".join(NAME.get(m, m) for m in models)
    body = ("\\begin{tabular}{lrrccccc}\n\\toprule\n"
            f"Fraction & Train & Val. & {head} \\\\\n\\midrule\n" + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}")
    _write("tab_dataeff",
           "Matched-budget data efficiency at week 2: the training and validation splits are subsampled at the "
           "same fraction (test ROC-AUC, mean $\\pm$ standard deviation over five seeds; Train and Val. are the "
           "mean numbers of enrolments).", "tab:dataeff", body, size="\\scriptsize")


def main() -> int:
    for fn in (cohort, predictive, sensitivity, aucdiff, fusioncoef, band, alerts, diagnostic, xai, earlywarning, calibration,
               fairness, generalization, dataeff):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {fn.__name__}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
