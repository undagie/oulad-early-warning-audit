"""Hasilkan figур paper dari CSV di outputs/. Lewati figур bila CSV-nya belum ada.

    python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("outputs")
FIG = OUT / "figures"


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    stem = Path(name).stem
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / name, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {FIG / name}")


def fig_auc_vs_window():
    f = OUT / "table1_predictive.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, g in t.groupby("model"):
        g = g.sort_values("window")
        ax.plot(g["window"], g["auc"], marker="o", label=model)
    ax.set_xlabel("Prediction window (week)")
    ax.set_ylabel("ROC-AUC (test)")
    ax.set_title("Predictive performance across windows")
    ax.set_xticks(sorted(t["window"].unique()))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "fig1_auc_vs_window.png")


def fig_coverage_vs_window():
    f = OUT / "table3_explanation.csv"
    if not f.exists():
        return
    t = pd.read_csv(f).sort_values("window")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t["window"], t["coverage"], marker="o", color="#2a7")
    ax.set_xlabel("Prediction window (week)")
    ax.set_ylabel("Explanation coverage (NS-GRU)")
    ax.set_title("Explanation coverage strengthens over time (RQ4)")
    ax.set_xticks(t["window"])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    _save(fig, "fig2_coverage_vs_window.png")


def fig_rule_diagnostic():
    f = OUT / "table3b_rule_diagnostic.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    w = d["window"].max()
    d = d[d["window"] == w].sort_values("precision", ascending=False)
    base = None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(d["rule"], d["precision"], color="#369")
    ax.axhline(0.42, ls="--", color="gray", label="base rate ~0.42")
    ax.set_ylabel("Precision = P(at-risk | rule fired)")
    ax.set_title(f"Rule diagnostic validity (Week {w})")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    _save(fig, "fig3_rule_diagnostic.png")


def fig_activation_heatmap():
    f = OUT / "rule_activation.csv"
    if not f.exists():
        return
    a = pd.read_csv(f)
    piv = a.pivot_table(index="rule", columns="window", values="activation", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.pcolormesh(
        range(len(piv.columns) + 1),
        range(len(piv.index) + 1),
        piv.values,
        cmap="YlOrRd",
        vmin=0,
        vmax=0.35,
        shading="flat",
    )
    ax.set_aspect("auto")
    ax.invert_yaxis()
    ax.set_xticks([j + 0.5 for j in range(len(piv.columns))])
    ax.set_xticklabels(piv.columns)
    ax.set_yticks([i + 0.5 for i in range(len(piv.index))])
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("Window (week)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j + 0.5, i + 0.5, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, ticks=[0.0, 0.1, 0.2, 0.3])
    cbar.solids.set_rasterized(False)
    cbar.set_label("Activation rate")
    _save(fig, "fig4_activation_heatmap.png")


def fig_strict_vs_leaky():
    f = OUT / "ablation_strict_vs_leaky.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    w = d["window"].min()
    d = d[d["window"] == w]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(d))
    ax.bar([i - 0.2 for i in x], d["auc_strict"], width=0.4, label="strict (LEAP)")
    ax.bar([i + 0.2 for i in x], d["auc_leaky"], width=0.4, label="leaky")
    ax.set_xticks(list(x))
    ax.set_xticklabels(d["model"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Leakage inflation (Week {w}): strict vs leaky")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    _save(fig, "fig5_strict_vs_leaky.png")


def main() -> int:
    print("Membuat figures ke outputs/figures/ ...")
    fig_auc_vs_window()
    fig_coverage_vs_window()
    fig_rule_diagnostic()
    fig_activation_heatmap()
    fig_strict_vs_leaky()
    print("Selesai.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
