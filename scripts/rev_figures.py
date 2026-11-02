"""Revisi SOIC 4706: figur naskah dari tabel outputs/revision/tables (vektor PDF + PNG 600 dpi).

  fig_temporal_validity  (A) AUC strict per minggu, (B) inflasi leaky - strict (CI 95% corrected t),
                         (C) perubahan AUC GRU akibat lag visibilitas skor
  fig_activation_heatmap aktivasi rule per minggu (test, lima seed)
  fig_operating_points   (A) precision & recall vs kapasitas (Week 8), (B) net benefit (Week 8)

    python scripts/rev_figures.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "revision" / "tables"
FIG = ROOT / "outputs" / "figures"
SUB = ROOT / "submission_soic"
matplotlib.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                            "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
COLORS = {"NS-GRU": "#DD8452", "GRU": "#4C72B0", "XGBoost": "#8172B3", "Static-only": "#7F7F7F",
          "Rules-only": "#55A868", "Treat all": "#BBBBBB"}
LABEL = {"NS-GRU-2in": "NS-GRU"}
WEEKS = [2, 4, 6, 8]


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    if SUB.is_dir():
        for ext in ("pdf", "png"):
            shutil.copy(FIG / f"{name}.{ext}", SUB / f"{name}.{ext}")
    print(f"  -> {name}")


def temporal_validity() -> None:
    rows = pd.read_csv(TAB / "main_per_seed.csv")
    sens = pd.read_csv(TAB / "sensitivity_vs_main.csv")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.6, 3.0))
    for m in ["NS-GRU-2in", "GRU", "XGBoost", "Rules-only", "Static-only"]:
        g = rows[rows.model == m].groupby("window").auc.agg(["mean", "std"]).reindex(WEEKS)
        name = LABEL.get(m, m)
        a.errorbar(WEEKS, g["mean"], yerr=g["std"], marker="o", ms=3, lw=1.1, capsize=2,
                   color=COLORS[name], label=name)
    fx = sens[(sens.condition == "fixed_cohort") & (sens.model == "NS-GRU-2in")].set_index("window").reindex(WEEKS)
    if fx.auc_mean.notna().any():
        a.plot(WEEKS, fx.auc_mean, ls="--", marker="o", ms=2.5, mfc="none", lw=0.9, color=COLORS["NS-GRU"],
               label="NS-GRU, fixed cohort")
    a.set(title="(A) Cutoff-first ROC-AUC", xlabel="Prediction week", ylabel="ROC-AUC (mean $\\pm$ SD)",
          xticks=WEEKS)
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=6)
    lk = sens[sens.condition == "leaky"]
    for i, m in enumerate(["NS-GRU-2in", "XGBoost", "Static-only"]):
        g = lk[lk.model == m].set_index("window").reindex(WEEKS)
        x = np.array(WEEKS) + (i - 1) * 0.35
        name = LABEL.get(m, m)
        b.errorbar(x, g.delta_mean, yerr=[g.delta_mean - g.delta_ci_low, g.delta_ci_high - g.delta_mean],
                   fmt="o", ms=3, capsize=2, color=COLORS[name], label=name)
    b.axhline(0, color="#999", lw=0.6)
    b.set(title="(B) Inflation when post-cutoff\nevidence is admitted", xlabel="Prediction week",
          ylabel="$\\Delta$ROC-AUC, leaky $-$ strict", xticks=WEEKS)
    b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=6)
    for lag, ls in (("lag7", ":"), ("lag14", "--"), ("lag21", "-")):
        g = sens[(sens.condition == lag) & (sens.model == "NS-GRU-2in")].set_index("window").reindex(WEEKS)
        c.errorbar(WEEKS, g.delta_mean, yerr=[g.delta_mean - g.delta_ci_low, g.delta_ci_high - g.delta_mean],
                   ls=ls, marker="o", ms=3, capsize=2, color=COLORS["NS-GRU"], lw=1, label=f"{lag[3:]}-day lag")
    c.axhline(0, color="#999", lw=0.6)
    c.set(title="(C) NS-GRU under delayed\nscore visibility", xlabel="Prediction week",
          ylabel="$\\Delta$ROC-AUC vs primary analysis", xticks=WEEKS)
    c.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=6)
    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig_temporal_validity")


def activation_heatmap() -> None:
    rm = pd.read_csv(TAB / "rule_metrics.csv")
    piv = rm.pivot(index="rule", columns="window", values="activation").reindex(columns=WEEKS)
    names = {"R1": "R1 Persistent inactivity", "R2": "R2 Low engagement", "R3": "R3 Missing resources",
             "R4": "R4 Poor early score", "R5": "R5 Late submission", "R6": "R6 Repeat + low engagement",
             "R7": "R7 Pre-deadline inactivity"}
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    im = ax.imshow(piv.values, cmap="YlOrRd", vmin=0, vmax=0.35, aspect="auto")
    ax.set_xticks(range(len(WEEKS)), [str(w) for w in WEEKS])
    ax.set_yticks(range(len(piv.index)), [names.get(r, r) for r in piv.index])
    ax.set_xlabel("Prediction week")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.05)
    cb.set_label("Activation rate")
    fig.tight_layout()
    _save(fig, "fig_activation_heatmap")


def operating() -> None:
    op = pd.read_csv(TAB / "operating_points.csv")
    nb = pd.read_csv(TAB / "net_benefit.csv")
    from matplotlib.lines import Line2D

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.55))
    w = 8
    # XGBoost first with large hollow markers, NS-GRU on top with small filled markers: the curves coincide
    style = {"XGBoost": dict(marker="o", ms=5, mfc="none", lw=1.6),
             "NS-GRU-2in": dict(marker="o", ms=2.6, lw=0.9),
             "Static-only": dict(marker="s", ms=2.8, lw=0.9)}
    for m in ["XGBoost", "NS-GRU-2in", "Static-only"]:
        g = op[(op.window == w) & (op.model == m)].groupby("capacity")[["precision", "recall"]].mean()
        col = COLORS[LABEL.get(m, m)]
        a.plot(g.index * 100, g.precision, color=col, **style[m])
        a.plot(g.index * 100, g.recall, color=col, ls="--", **style[m])
    a.set(title="(A) Capacity-constrained flagging", xlabel="Share of enrolments flagged (%)",
          ylabel="Precision (solid) or recall (dashed)", ylim=(0, 1.02), xticks=[5, 10, 20, 30])
    for m in ["XGBoost", "NS-GRU-2in", "Static-only", "Treat all"]:
        g = nb[(nb.window == w) & (nb.model == m)].groupby("threshold").net_benefit.mean()
        col = COLORS[LABEL.get(m, m)]
        st = style.get(m, dict(marker="^", ms=2.8, lw=0.9))
        b.plot(g.index, g.values, color=col, ls="--" if m == "Treat all" else "-", **st)
    b.axhline(0, color="#777", lw=0.7, ls=":")
    b.set(title="(B) Decision-curve net benefit", xlabel="Threshold probability", ylabel="Net benefit",
          ylim=(-0.2, 0.36), xticks=[0.2, 0.3, 0.4, 0.5, 0.6])
    handles = [Line2D([], [], color=COLORS["NS-GRU"], marker="o", ms=2.6, lw=0.9, label="NS-GRU"),
               Line2D([], [], color=COLORS["XGBoost"], marker="o", ms=5, mfc="none", lw=1.6, label="XGBoost"),
               Line2D([], [], color=COLORS["Static-only"], marker="s", ms=2.8, lw=0.9, label="Static-only"),
               Line2D([], [], color=COLORS["Treat all"], marker="^", ms=2.8, lw=0.9, ls="--", label="Treat all"),
               Line2D([], [], color="#777", lw=0.7, ls=":", label="Treat none")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=1.8, rect=(0, 0.08, 1, 1))
    _save(fig, "fig_operating_points")


def main() -> int:
    temporal_validity()
    activation_heatmap()
    operating()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
