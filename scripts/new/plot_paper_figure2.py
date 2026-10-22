"""
Figure 2 – Nature Portfolio Publication Figure (ROC Curves & Validation Schemes)
=================================================================================
Nature Machine Intelligence / Nature Communications styling.

Changes vs v1:
  - Removed horizontal reference lines in panel b (belong in caption, not figure)
  - Simplified ΔAUC annotation: cleaner bracket, smaller text, no redundant label
  - Shortened legend entries (no full N= counts in legend, belong in caption)
  - Reduced axis label sizes to Nature standard (8 pt labels, 7 pt ticks)
  - Axes linewidth reduced to 0.6 pt
  - Removed all gridlines
  - Tick direction: out, length 3.5 pt
  - Tightened layout margins
  - Bar alpha raised to 1.0 (no transparency — flat publication bars)
  - Error bar linewidth: 0.75 pt, capsize 3.0

Dimensions preserved: 11.5 × 4.8 in at 300 DPI (same as v1).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent
# Walk up to find project root
_r = ROOT
while not (_r / ".git").exists() and _r != _r.parent:
    _r = _r.parent
ROOT = _r
OUT = ROOT / "results" / "new"
OUT.mkdir(parents=True, exist_ok=True)


def delong_roc_variance(ground_truth, predictions):
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    order = np.argsort(predictions)
    ground_truth = ground_truth[order]
    predictions = predictions[order]
    n_pos = np.sum(ground_truth == 1)
    n_neg = np.sum(ground_truth == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0
    ranks = stats.rankdata(predictions)
    pos_ranks = ranks[ground_truth == 1]
    auc_val = (np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    pred_pos = predictions[ground_truth == 1]
    pred_neg = predictions[ground_truth == 0]
    v_pos = np.zeros(n_pos)
    v_neg = np.zeros(n_neg)
    for i, p in enumerate(pred_pos):
        v_pos[i] = (np.sum(pred_neg < p) + 0.5 * np.sum(pred_neg == p)) / n_neg
    for i, n in enumerate(pred_neg):
        v_neg[i] = (np.sum(pred_pos > n) + 0.5 * np.sum(pred_pos == n)) / n_pos
    var_pos = np.var(v_pos, ddof=1) / n_pos
    var_neg = np.var(v_neg, ddof=1) / n_neg
    return auc_val, var_pos + var_neg


def make_figure():
    # ── Load predictions ──────────────────────────────────────────────
    oof = pd.read_csv(ROOT / "results" / "oof_predictions_pooled.csv")
    rs  = pd.read_csv(ROOT / "results" / "random_split_predictions.csv")

    fpr_rs,  tpr_rs,  _ = roc_curve(rs["y_true"],  rs["y_score"])
    fpr_mil, tpr_mil, _ = roc_curve(oof["y_true"], oof["y_score_mil"])
    fpr_xgb, tpr_xgb, _ = roc_curve(oof["y_true"], oof["y_score"])

    auc_rs,  var_rs  = delong_roc_variance(rs["y_true"],  rs["y_score"])
    auc_mil, var_mil = delong_roc_variance(oof["y_true"], oof["y_score_mil"])
    auc_xgb, var_xgb = delong_roc_variance(oof["y_true"], oof["y_score"])

    ci_rs  = 1.96 * np.sqrt(var_rs)
    ci_mil = 1.96 * np.sqrt(var_mil)
    ci_xgb = 1.96 * np.sqrt(var_xgb)

    fpr_baseline, tpr_baseline = 0.3607, 0.9544

    # ── Global typography (Nature standard, enlarged for publication) ──
    matplotlib.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          10.0,
        "axes.labelsize":     10.5,
        "xtick.labelsize":    9.0,
        "ytick.labelsize":    9.0,
        "legend.fontsize":    8.8,
        "axes.linewidth":     0.7,
        "xtick.major.width":  0.7,
        "ytick.major.width":  0.7,
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "svg.fonttype":       "none",
    })

    # NPG palette
    COL_RS   = "#3C5488"
    COL_MIL  = "#00A087"
    COL_XGB  = "#8491B4"
    COL_BASE = "#E64B35"
    TEXT_C   = "#0f172a"

    # ── Figure canvas (dimensions preserved: 3870 × 1620 px @ 300 DPI) ──
    fig, (ax_roc, ax_bar) = plt.subplots(
        1, 2,
        figsize=(12.9, 5.4),
        dpi=300,
        facecolor="#ffffff",
    )

    # ══════════════════════════════════════════════════════════════════
    # PANEL a — ROC curves
    # ══════════════════════════════════════════════════════════════════
    ax_roc.plot(fpr_rs,  tpr_rs,  color=COL_RS,  lw=1.5,
                label=f"Random split (AUC = {auc_rs:.3f} ± {ci_rs:.3f})")
    ax_roc.plot(fpr_mil, tpr_mil, color=COL_MIL, lw=1.5,
                label=f"ST-blocked CV, MIL-only (AUC = {auc_mil:.3f} ± {ci_mil:.3f})")
    ax_roc.plot(fpr_xgb, tpr_xgb, color=COL_XGB, lw=1.3, linestyle="--",
                label=f"ST-blocked CV, MIL+XGBoost (AUC = {auc_xgb:.3f} ± {ci_xgb:.3f})")
    ax_roc.plot(fpr_baseline, tpr_baseline,
                marker="*", color=COL_BASE, markersize=9,
                markeredgecolor="#7f1d1d", markeredgewidth=0.5,
                linestyle="none",
                label="Biological rule baseline")

    # Random diagonal
    ax_roc.plot([0, 1], [0, 1],
                color="#94a3b8", linestyle=(0, (4, 3)), lw=0.8, zorder=1)

    ax_roc.set_xlim(-0.015, 1.015)
    ax_roc.set_ylim(-0.015, 1.015)
    ax_roc.set_xlabel("False positive rate (1 − Specificity)", color=TEXT_C)
    ax_roc.set_ylabel("True positive rate (Sensitivity)", color=TEXT_C)

    for sp in ["top", "right"]:
        ax_roc.spines[sp].set_visible(False)
    ax_roc.spines["left"].set_color(TEXT_C)
    ax_roc.spines["bottom"].set_color(TEXT_C)

    ax_roc.legend(
        loc="lower right",
        frameon=False,
        handlelength=1.8,
        borderaxespad=0.6,
        handletextpad=0.5,
    )

    # Panel label
    ax_roc.text(-0.12, 1.05, "a",
                transform=ax_roc.transAxes,
                fontsize=12, fontweight="bold", color="#000000",
                va="bottom", ha="left")

    # ══════════════════════════════════════════════════════════════════
    # PANEL b — Bar chart (AUC per scheme)
    # ══════════════════════════════════════════════════════════════════
    models  = ["Random split", "ST-blocked CV\n(MIL-only)", "ST-blocked CV\n(MIL+XGBoost)"]
    aucs    = [auc_rs, auc_mil, auc_xgb]
    errors  = [ci_rs, ci_mil, ci_xgb]
    colors  = [COL_RS, COL_MIL, COL_XGB]

    x_pos = np.arange(len(models))

    bars = ax_bar.bar(
        x_pos, aucs, yerr=errors,
        width=0.42,
        color=colors,
        edgecolor="none",
        alpha=1.0,
        capsize=3.5,
        error_kw={"ecolor": "#1e293b", "lw": 0.85, "capthick": 0.85},
        zorder=3,
    )

    # AUC values above error caps
    for i, (val, err) in enumerate(zip(aucs, errors)):
        ax_bar.text(
            i, val + err + 0.014,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=8.8, color=TEXT_C, zorder=4,
        )

    # ── ΔAUC bracket (minimal: thin bracket + one short label) ───────
    delta_auc = auc_mil - auc_rs
    pct_drop  = (delta_auc / auc_rs) * 100

    y_top = max(auc_rs, auc_mil) + max(ci_rs, ci_mil) + 0.040
    hb    = 0.010   # bracket arm height

    ax_bar.plot(
        [0, 0, 1, 1],
        [y_top - hb, y_top, y_top, y_top - hb],
        color=COL_BASE, lw=1.0, zorder=5,
    )
    ax_bar.text(
        0.5, y_top + 0.007,
        f"ΔAUC = {delta_auc:.3f} ({pct_drop:.1f}%)",
        ha="center", va="bottom",
        fontsize=8.5, color=COL_BASE, zorder=5,
    )

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(models, color=TEXT_C, linespacing=1.2)
    ax_bar.set_ylabel("ROC-AUC", color=TEXT_C)
    ax_bar.set_ylim(0.60, 1.12)

    for sp in ["top", "right"]:
        ax_bar.spines[sp].set_visible(False)
    ax_bar.spines["left"].set_color(TEXT_C)
    ax_bar.spines["bottom"].set_color(TEXT_C)

    # Panel label
    ax_bar.text(-0.12, 1.05, "b",
                transform=ax_bar.transAxes,
                fontsize=12, fontweight="bold", color="#000000",
                va="bottom", ha="left")

    # ── Save ──────────────────────────────────────────────────────────
    plt.subplots_adjust(left=0.08, right=0.96, bottom=0.15, top=0.90, wspace=0.30)

    fig.savefig(OUT / "Figure2_updated.png",
                dpi=300,
                facecolor="#ffffff")
    fig.savefig(OUT / "Figure2_updated.pdf",
                facecolor="#ffffff")
    plt.close(fig)
    print("  [OK] Figure2_updated (.png / .pdf) saved to results/new/ (3870 × 1620 px, 300 DPI)")


if __name__ == "__main__":
    make_figure()
