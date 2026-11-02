"""Figur tambahan dari artefak prediksi (butuh run_extra_metrics.py lebih dulu):
  fig8 ROC curves, fig9 PR curves (per window, model utama),
  fig10 confusion matrices (Week 8), fig11 detection curve, fig12 Jaccard stability.

    python scripts/make_extra_figures.py            # split student
    python scripts/make_extra_figures.py --gen      # generalisasi 2013->2014
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)

OUT = Path("outputs")
FIG = OUT / "figures"
CURVE_MODELS = ["NS-GRU", "GRU", "XGBoost", "Static-only"]


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print(f"  -> {FIG / name}")


def _one_seed(preds: pd.DataFrame) -> pd.DataFrame:
    """Pakai satu seed (terkecil) untuk kurva agar tak mencampur run."""
    s = sorted(preds["seed"].unique())[0]
    return preds[preds["seed"] == s]


def fig_roc(preds: pd.DataFrame, sfx: str) -> None:
    windows = sorted(preds["window"].unique())
    fig, axes = plt.subplots(1, len(windows), figsize=(4 * len(windows), 4), squeeze=False)
    for ax, k in zip(axes[0], windows):
        gk = preds[preds["window"] == k]
        for model in CURVE_MODELS:
            gm = gk[gk["model"] == model]
            if gm.empty:
                continue
            fpr, tpr, _ = roc_curve(gm["y_true"], gm["p"])
            a = roc_auc_score(gm["y_true"], gm["p"])
            ax.plot(fpr, tpr, label=f"{model} ({a:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=0.8)
        ax.set_title(f"Week {k}")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)
    fig.suptitle("ROC curves per window" + ("" if not sfx else " (generalization)"))
    _save(fig, f"fig8_roc_curves{sfx}.png")


def fig_pr(preds: pd.DataFrame, sfx: str) -> None:
    windows = sorted(preds["window"].unique())
    fig, axes = plt.subplots(1, len(windows), figsize=(4 * len(windows), 4), squeeze=False)
    for ax, k in zip(axes[0], windows):
        gk = preds[preds["window"] == k]
        base = float(gk[gk["model"] == "GRU"]["y_true"].mean()) if not gk.empty else 0.0
        for model in CURVE_MODELS:
            gm = gk[gk["model"] == model]
            if gm.empty:
                continue
            prec, rec, _ = precision_recall_curve(gm["y_true"], gm["p"])
            ap = average_precision_score(gm["y_true"], gm["p"])
            ax.plot(rec, prec, label=f"{model} ({ap:.3f})")
        ax.axhline(base, ls="--", color="gray", lw=0.8, label=f"base {base:.2f}")
        ax.set_title(f"Week {k}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_ylim(0, 1); ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=0.3)
    fig.suptitle("Precision-Recall curves per window" + ("" if not sfx else " (generalization)"))
    _save(fig, f"fig9_pr_curves{sfx}.png")


def fig_confusion(sfx: str) -> None:
    f = OUT / f"table4_confusion{sfx}.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    w = d["window"].max()
    d = d[d["window"] == w]
    models = ["NS-GRU", "GRU", "XGBoost", "Static-only"]
    d = d[d["model"].isin(models)]
    fig, axes = plt.subplots(1, len(d), figsize=(3 * len(d), 3), squeeze=False)
    for ax, (_, r) in zip(axes[0], d.iterrows()):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], dtype=float)
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"], fontsize=8)
        ax.set_title(f"{r['model']}\nF1={r['f1']:.3f}", fontsize=9)
    fig.suptitle(f"Confusion matrices (Week {w})")
    _save(fig, f"fig10_confusion{sfx}.png")


def fig_detection_curve(sfx: str) -> None:
    f = OUT / f"detection_curve{sfx}.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, g in d.groupby("model"):
        g = g.sort_values("window")
        ax.plot(g["window"], g["cum_detect_frac"], marker="o", label=model)
    ax.set_xlabel("Prediction window (week)")
    ax.set_ylabel("Cumulative fraction of at-risk detected (top-10%)")
    ax.set_title("Detection lead time: earlier is better")
    ax.set_xticks(sorted(d["window"].unique()))
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save(fig, f"fig11_detection_curve{sfx}.png")


def fig_jaccard(sfx: str) -> None:
    f = OUT / f"jaccard_stability{sfx}.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(d["window_pair"], d["jaccard_mean"],
           yerr=d.get("jaccard_std"), capsize=4, color="#5a3")
    ax.set_ylabel("Mean Jaccard(rule set)")
    ax.set_xlabel("Adjacent window pair")
    ax.set_title("Explanation stability across windows")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3, axis="y")
    _save(fig, f"fig12_jaccard{sfx}.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true", help="pakai artefak generalisasi (_gen)")
    args = ap.parse_args()
    sfx = "_gen" if args.gen else ""

    pred_p = OUT / f"test_predictions{sfx}.csv"
    print(f"Membuat figur tambahan (sfx='{sfx or 'student'}') ...")
    if pred_p.exists():
        preds = _one_seed(pd.read_csv(pred_p))
        fig_roc(preds, sfx)
        fig_pr(preds, sfx)
    else:
        print(f"  (lewati ROC/PR: {pred_p} belum ada — jalankan run_extra_metrics.py)")
    fig_confusion(sfx)
    fig_detection_curve(sfx)
    fig_jaccard(sfx)
    print("Selesai.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
