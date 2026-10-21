"""
Figure 6 – Nature Portfolio Publication Figure (Precision-Recall & Calibration Analysis)
========================================================================================
Nature Portfolio styling (Nature Machine Intelligence / Nature Communications):
  - Strict 2-panel layout:
      * Panel a: Precision-Recall (PR) curve comparing model performance to empirical baseline prevalence.
      * Panel b: Probability calibration curve (reliability diagram) with binned empirical proportions.
  - 100% Real Empirical Data:
      * Evaluated on held-out test cohort (results/random_split_predictions.csv, N = 770 isolates).
      * Average Precision (AP): 0.9470 (prevalence = 0.3766).
      * Brier Score Loss: 0.0739 across 10 calibration bins.
  - Nature Publishing Group (NPG) Palette:
      * PR curve: NPG Classic Navy (#3C5488)
      * Calibration curve: NPG Coral Red (#E64B35)
      * Reference baselines: Muted Slate Gray (#94A3B8)
  - Typography & Despining:
      * Axis labels: 8.0 pt; tick labels: 7.0 pt; legend: 7.0 pt.
      * Axes linewidth: 0.6 pt; ticks outward (3.0 pt).
      * Top and right spines removed; no canvas/subplot titles.
      * Panel labels: bold lowercase **a** and **b** (10 pt).
  - Dual export: 300 DPI PNG & vector TrueType PDF (pdf.fonttype = 42).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
while not (ROOT / ".git").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

OUT = ROOT / "results" / "new"
OUT.mkdir(parents=True, exist_ok=True)


def make_figure():
    # ── 1. Load real empirical predictions ────────────────────────────
    pred_path = ROOT / "results" / "random_split_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    df = pd.read_csv(pred_path)
    y_true = df["y_true"].values
    y_score = df["y_score"].values

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)
    brier = brier_score_loss(y_true, y_score)
    prevalence = y_true.mean()

    # ── 2. Nature typography ──────────────────────────────────────────
    matplotlib.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          8.0,
        "axes.labelsize":     8.0,
        "xtick.labelsize":    7.0,
        "ytick.labelsize":    7.0,
        "legend.fontsize":    7.0,
        "axes.linewidth":     0.6,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.major.size":   3.0,
        "ytick.major.size":   3.0,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "svg.fonttype":       "none",
        "figure.dpi":         300,
    })

    # Colors
    COL_NAVY  = "#3C5488"
    COL_CORAL = "#E64B35"
    COL_GRAY  = "#94A3B8"
    TEXT_C    = "#0f172a"

    # ── 3. Canvas ─────────────────────────────────────────────────────
    fig, (ax_pr, ax_cal) = plt.subplots(
        1, 2,
        figsize=(8.5, 3.8),
        dpi=300,
        facecolor="#ffffff"
    )

    # ══════════════════════════════════════════════════════════════════
    # PANEL a — Precision-Recall Curve
    # ══════════════════════════════════════════════════════════════════
    ax_pr.plot(
        rec, prec,
        color=COL_NAVY, lw=1.5,
        label=f"MIL v5.3 (AP = {ap:.3f})"
    )
    ax_pr.axhline(
        prevalence,
        color=COL_GRAY, linestyle=(0, (4, 3)), lw=0.9,
        label=f"Baseline prevalence ({prevalence:.3f})"
    )

    ax_pr.set_xlim(-0.015, 1.015)
    ax_pr.set_ylim(-0.015, 1.015)
    ax_pr.set_xlabel("Recall (Sensitivity)", color=TEXT_C)
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", color=TEXT_C)

    for sp in ["top", "right"]:
        ax_pr.spines[sp].set_visible(False)
    ax_pr.spines["left"].set_color(TEXT_C)
    ax_pr.spines["bottom"].set_color(TEXT_C)

    ax_pr.legend(loc="lower left", frameon=False, handlelength=1.6)

    # Panel label
    ax_pr.text(-0.16, 1.06, "a",
               transform=ax_pr.transAxes,
               fontsize=10, fontweight="bold", color="#000000",
               va="bottom", ha="left")

    # ══════════════════════════════════════════════════════════════════
    # PANEL b — Probability Calibration Curve (Reliability Diagram)
    # ══════════════════════════════════════════════════════════════════
    ax_cal.plot(
        prob_pred, prob_true,
        marker="o", markersize=4.5, markeredgewidth=0.5, markeredgecolor="#7f1d1d",
        color=COL_CORAL, lw=1.4,
        label=f"MIL v5.3 (Brier = {brier:.4f})"
    )
    ax_cal.plot(
        [0, 1], [0, 1],
        color=COL_GRAY, linestyle=(0, (4, 3)), lw=0.9,
        label="Perfect calibration"
    )

    ax_cal.set_xlim(-0.015, 1.015)
    ax_cal.set_ylim(-0.015, 1.015)
    ax_cal.set_xlabel("Predicted probability of resistance", color=TEXT_C)
    ax_cal.set_ylabel("Observed fraction of resistant isolates", color=TEXT_C)

    for sp in ["top", "right"]:
        ax_cal.spines[sp].set_visible(False)
    ax_cal.spines["left"].set_color(TEXT_C)
    ax_cal.spines["bottom"].set_color(TEXT_C)

    ax_cal.legend(loc="upper left", frameon=False, handlelength=1.6)

    # Panel label
    ax_cal.text(-0.16, 1.06, "b",
                transform=ax_cal.transAxes,
                fontsize=10, fontweight="bold", color="#000000",
                va="bottom", ha="left")

    # ── 4. Save ───────────────────────────────────────────────────────
    plt.subplots_adjust(left=0.09, right=0.96, bottom=0.15, top=0.90, wspace=0.28)

    fig.savefig(OUT / "Figure6_updated.png",
                dpi=300, bbox_inches="tight", pad_inches=0.08,
                facecolor="#ffffff")
    fig.savefig(OUT / "Figure6_updated.pdf",
                bbox_inches="tight", pad_inches=0.08,
                facecolor="#ffffff")
    plt.close(fig)
    print(f"  [OK] Figure6_updated (.png / .pdf) saved to results/new/ (AP={ap:.4f}, Brier={brier:.4f})")


if __name__ == "__main__":
    make_figure()
