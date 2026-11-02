"""Diagram arsitektur NS-GRU untuk paper (Figure 1).

Blok: input cutoff-constrained (LEAP) -> cabang neural (GRU) + cabang symbolic (7 rules)
-> standardized logistic fusion -> skor at-risk + rule-based alerts yang dapat diaudit.

Digambar langsung pada ukuran cetak (lebar = \\textwidth iapress, 6.5 in), sehingga ukuran
font di sini = ukuran font di PDF (target >= 7.5 pt).

    python scripts/make_architecture.py   -> outputs/figures/fig_architecture.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FIG = Path("outputs/figures")

# Lebar teks iapress.cls = 6.5 in; 1 unit data = 0.5 in.
WIDTH_IN, HEIGHT_IN = 6.5, 4.45
FS = 8.0          # teks kotak
FS_LABEL = 7.5    # label panah
FS_GROUP = 8.0    # judul grup

C_NEURAL = "#cfe0f3"; E_NEURAL = "#2f6db5"
C_SYM = "#fbe2c4"; E_SYM = "#d1802a"
C_FUSION = "#e2d3f0"; E_FUSION = "#7a4fa0"
C_OUT = "#cdeccd"; E_OUT = "#3a8a3a"
C_IN = "#eceff2"; E_IN = "#7a8896"


def box(ax, x, y, w, h, text, fc, ec, fs=FS, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#1a1a1a", linespacing=1.25)


def arrow(ax, p0, p1, label=None, color="#444", style="-", rad=0.0, dy=0.12):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                                 lw=1.0, color=color, linestyle=style,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + dy, label, ha="center", va="bottom", fontsize=FS_LABEL,
                style="italic", color=color)


def group(ax, x, y, w, h, title, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="none", ec=ec, lw=0.9, linestyle=(0, (4, 3))))
    ax.text(x + 0.15, y + h - 0.12, title, ha="left", va="top", fontsize=FS_GROUP,
            color=ec, fontweight="bold")


def main() -> int:
    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 2 * WIDTH_IN); ax.set_ylim(0, 2 * HEIGHT_IN); ax.axis("off")

    # Banner LEAP
    ax.add_patch(FancyBboxPatch((0.1, 8.15), 12.8, 0.6, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="#fff6e6", ec="#c9a227", lw=0.9))
    ax.text(6.5, 8.45, r"LEAP: every input truncated to the end of week $k$ before any join",
            ha="center", va="center", fontsize=8.5, color="#7a5c00", fontweight="bold")

    # --- Kolom input ---
    box(ax, 0.1, 5.55, 2.8, 1.35, "Weekly VLE sequence\n$x^{seq}\\in\\mathbb{R}^{K\\times 12}$", C_IN, E_IN)
    box(ax, 0.1, 3.55, 2.8, 1.35, "Tabular features\n(static, assessment,\ncourse structure)", C_IN, E_IN)
    box(ax, 0.1, 1.15, 2.8, 1.35, "Rule frame\n(engagement,\nassessment, deadline)", C_IN, E_IN)

    # --- Cabang neural ---
    group(ax, 3.25, 5.05, 5.1, 2.75, "Neural branch", E_NEURAL)
    box(ax, 3.45, 5.3, 2.15, 1.75, "GRU encoder\nstate $h$ (size $d$)\n$d$ tuned per seed",
        C_NEURAL, E_NEURAL)
    box(ax, 6.05, 5.3, 2.15, 1.75, "$[h;\\,x^{tab}]\\to$ Dense\n$\\to$ Dropout $\\to\\sigma$\n$\\Rightarrow P_{neural}$",
        C_NEURAL, E_NEURAL)

    # --- Cabang symbolic ---
    group(ax, 3.25, 0.25, 5.1, 4.2, "Symbolic branch", E_SYM)
    box(ax, 3.45, 1.4, 2.15, 1.75, "Rule engine\n7 alerts $R_1$ to $R_7$\n(train thresholds)", C_SYM, E_SYM)
    box(ax, 6.05, 2.55, 2.15, 1.25, "$P_{sym}$\n(weighted alerts)", C_SYM, E_SYM)
    box(ax, 6.05, 0.5, 2.15, 1.45, "Alert texts\ncondition + review\naction per rule", C_OUT, E_OUT)

    # --- Fusion + output ---
    box(ax, 9.1, 3.9, 3.8, 2.05,
        "Standardized logistic fusion\n$\\sigma(\\gamma_0+\\gamma_1 z(P_{neural})$\n$+\\gamma_2 z(P_{sym}))$\n$\\Rightarrow P_{final}$",
        C_FUSION, E_FUSION)
    box(ax, 9.1, 1.0, 3.8, 1.6, "At-risk score +\nrule-based alerts", C_OUT, E_OUT, bold=True)

    # --- Panah ---
    arrow(ax, (2.9, 6.2), (3.45, 6.2))                                    # seq -> GRU
    ax.plot([2.9, 7.1], [4.62, 4.62], color="#444", lw=1.0)             # tab -> dense (siku)
    arrow(ax, (7.1, 4.62), (7.1, 5.3))
    ax.text(5.0, 4.66, "$x^{tab}$", ha="center", va="bottom", fontsize=FS_LABEL,
            style="italic", color="#444")
    arrow(ax, (2.9, 1.8), (3.45, 2.2))                                    # rule frame -> engine
    arrow(ax, (5.6, 6.2), (6.05, 6.2))                                    # GRU -> dense
    ax.text(5.83, 6.35, "$h$", ha="center", va="bottom", fontsize=FS_LABEL, style="italic", color="#444")
    arrow(ax, (5.6, 2.6), (6.05, 3.1))                                    # engine -> sym feats
    arrow(ax, (5.6, 1.9), (6.05, 1.3))                                    # engine -> alert texts
    arrow(ax, (8.2, 6.2), (9.3, 5.95), color=E_NEURAL, rad=-0.1)
    ax.text(8.5, 6.3, "$P_{neural}$", ha="left", va="bottom", fontsize=FS_LABEL, color=E_NEURAL)
    arrow(ax, (8.2, 3.2), (9.3, 3.9), color=E_SYM, rad=0.1)
    ax.text(9.35, 3.45, "$P_{sym}$", ha="left", va="center", fontsize=FS_LABEL,
            style="italic", color=E_SYM)
    arrow(ax, (10.95, 3.9), (10.95, 2.6), color=E_FUSION)                 # fusion -> output
    arrow(ax, (8.2, 1.2), (9.1, 1.5), color=E_OUT, style=(0, (3, 2)), rad=0.05)
    ax.text(8.75, 1.55, "audit", ha="center", va="bottom", fontsize=FS_LABEL,
            style="italic", color=E_OUT)

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_architecture.pdf")
    fig.savefig(FIG / "fig_architecture.png", dpi=600)
    plt.close(fig)
    print(f"-> {FIG/'fig_architecture.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
