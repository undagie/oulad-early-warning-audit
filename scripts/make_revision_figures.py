"""Figures for the revision, generated only from existing result tables:
  fig_calibration      — validation-fit calibration summaries with seed-level uncertainty.
  fig_temporal_leakage — fixed-cohort temporal-validity boundary test.
  fig_decile_map.png   — complementarity map: across P_neural deciles, the observed base rate and
                          the mean triggered-rule count rise in lockstep => rules fire where the
                          neural model is already confident (redundancy).

Copies both into submission_soic/ for the LaTeX build.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
SUB = ROOT / "submission_soic"
C = {"GRU": "#4C72B0", "NS-GRU": "#DD8452", "NS-GRU-cal": "#55A868", "XGBoost": "#8172B3"}
matplotlib.rcParams["pdf.fonttype"] = 42


def _to_manuscript(name: str) -> None:
    """Salin figur ke folder naskah LaTeX bila folder itu ada (tidak ada di repo publik)."""
    if SUB.is_dir():
        shutil.copy(FIG / name, SUB / name)
matplotlib.rcParams["ps.fonttype"] = 42


def fig_calibration():
    tab = pd.read_csv(OUT / "table_calibration.csv")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))
    windows = [2, 4, 6, 8]
    models = ["GRU", "NS-GRU", "NS-GRU-cal", "XGBoost"]
    markers = {"GRU": "o", "NS-GRU": "s", "NS-GRU-cal": "^", "XGBoost": "D"}
    styles = {"GRU": "-", "NS-GRU": "--", "NS-GRU-cal": "-.", "XGBoost": ":"}
    for m in models:
        d = tab[tab.model == m].set_index("window").loc[windows]
        axA.errorbar(windows, d.ece_mean, yerr=d.ece_std, marker=markers[m], ls=styles[m],
                     color=C[m], capsize=3, lw=1.8, label=m)
        axB.errorbar(windows, d.brier_mean, yerr=d.brier_std, marker=markers[m], ls=styles[m],
                     color=C[m], capsize=3, lw=1.8, label=m)
    for ax in (axA, axB):
        ax.set_xticks(windows); ax.grid(alpha=0.3); ax.set_xlabel("Prediction window (week)")
    axA.set_ylabel("ECE (lower is better)"); axA.set_title("(A) Expected calibration error")
    axB.set_ylabel("Brier score (lower is better)"); axB.set_title("(B) Brier score")
    axA.legend(fontsize=8)

    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / "fig_calibration.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_calibration.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    _to_manuscript("fig_calibration.pdf")
    _to_manuscript("fig_calibration.png")
    print("  -> fig_calibration.png")


def fig_temporal_leakage():
    d = pd.read_csv(OUT / "ablation_strict_vs_leaky.csv")
    models = ["NS-GRU", "GRU", "XGBoost", "Static-only"]
    colors = {"NS-GRU": "#0072B2", "GRU": "#D55E00", "XGBoost": "#009E73", "Static-only": "#666666"}
    markers = {"NS-GRU": "o", "GRU": "s", "XGBoost": "^", "Static-only": "D"}
    styles = {"NS-GRU": "-", "GRU": "--", "XGBoost": "-.", "Static-only": ":"}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))
    for m in models:
        g = d[d.model == m].sort_values("window")
        axA.plot(g.window, g.auc_strict, marker=markers[m], ls=styles[m], color=colors[m], lw=2, label=m)
        axB.plot(g.window, g.inflation, marker=markers[m], ls=styles[m], color=colors[m], lw=2, label=m)
    for ax in (axA, axB):
        ax.set_xticks([2, 4, 6, 8]); ax.set_xlabel("Prediction window (week)"); ax.grid(alpha=0.3)
    axA.set_ylabel("Strict ROC-AUC"); axA.set_ylim(0.58, 0.86)
    axA.set_title("(A) Cutoff-constrained performance")
    axB.axhline(0, color="#888888", lw=1)
    axB.set_ylabel("ROC-AUC inflation"); axB.set_ylim(-0.01, 0.25)
    axB.set_title("(B) Inflation from post-cutoff evidence")
    handles, labels = axA.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(FIG / "fig_temporal_leakage.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_temporal_leakage.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    _to_manuscript("fig_temporal_leakage.pdf")
    _to_manuscript("fig_temporal_leakage.png")
    print("  -> fig_temporal_leakage.pdf")


def fig_decile_map():
    dm = pd.read_csv(OUT / "diagnosis_decile_map.csv")
    w = 8
    d = dm[dm.window == w].sort_values("decile")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(d.decile, d.base_rate, "o-", color="#C44E52", lw=2, label="Observed at-risk rate")
    ax.plot(d.decile, d.mean_pneural, "s--", color=C["GRU"], lw=2, label=r"Mean $P_{\mathrm{neural}}$")
    ax.set_xlabel(r"$P_{\mathrm{neural}}$ decile (low $\to$ high risk)")
    ax.set_ylabel("Probability / rate")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(d.decile, d.mean_rulecount, "^-", color="#55A868", lw=2, label="Mean triggered-rule count")
    ax2.set_ylabel("Mean triggered-rule count", color="#55A868")
    ax2.tick_params(axis="y", labelcolor="#55A868")
    ax2.legend(loc="lower right", fontsize=9)
    ax.set_title("Rules fire where the neural model is already confident (week 8)")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / "fig_decile_map.png", dpi=150); plt.close(fig)
    _to_manuscript("fig_decile_map.png")
    print("  -> fig_decile_map.png")


if __name__ == "__main__":
    fig_calibration()
    fig_temporal_leakage()
    fig_decile_map()
