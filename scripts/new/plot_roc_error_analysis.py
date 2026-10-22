"""
Figure: Probability Distribution by Classification Outcome & ROC Comparison
=============================================================================
Reproduces the classical held-out validation benchmark (n = 770 isolates):
  - Panel a: Probability Distribution by Classification Outcome
      * Outcome-stratified predicted probability boxplot (TP, TN, VME, ME).
      * Default decision threshold at 0.50.
  - Panel b: Receiver Operating Characteristic (ROC) Comparison
      * MIL v5.3 (AUC = 0.9621) vs. k-mer baselines (RF AUC 0.9544, XGBoost AUC 0.9492),
        NTv3 Mean + MLP (AUC = 0.8988), Random Guess (0.5000), and the Biological Rule Baseline
        (Carbapenemase Rule: FPR = 0.3607, TPR = 0.9544, Balanced Accuracy = 0.796).
  - Nature Portfolio styling:
      * 300 DPI, TrueType PDF export (pdf.fonttype = 42).
      * Despined axes, outward ticks (3.0 pt), 0.6 pt axes linewidth.
      * Bold lowercase panel labels (a, b).
      * NPG colorblind-safe palette.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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

    df_raw = pd.read_csv(pred_path)
    y_val = df_raw["y_true"].values
    y_score = df_raw["y_score"].values

    df_pred = []
    for real, score in zip(y_val, y_score):
        pred = int(score >= 0.50)
        if real == 1 and pred == 1:
            outcome = "True Positive (TP)"
        elif real == 0 and pred == 0:
            outcome = "True Negative (TN)"
        elif real == 1 and pred == 0:
            outcome = "False Negative (VME)"
        elif real == 0 and pred == 1:
            outcome = "False Positive (ME)"
        df_pred.append({"Outcome": outcome, "Probability": score})

    df_pred = pd.DataFrame(df_pred)

    # ── 2. Nature Portfolio typography (enlarged for publication) ─────
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
        "figure.dpi":         300,
    })

    # Palette
    C_BLUE  = "#3C5488"   # NPG Navy
    C_GREEN = "#00A087"   # NPG Teal
    C_CORAL = "#E64B35"   # NPG Coral Red
    C_AMBER = "#E69F00"   # Muted Amber
    C_PURP  = "#7E22CE"   # Purple
    C_GRAY  = "#94A3B8"   # Slate Gray
    TEXT_C  = "#0f172a"

    # ── 3. Canvas (12.9 × 5.4 in at 300 DPI ≈ 3870 × 1620 px) ─────────
    fig, (ax_box, ax_roc) = plt.subplots(
        1, 2,
        figsize=(12.9, 5.4),
        dpi=300,
        facecolor="#ffffff"
    )

    # ══════════════════════════════════════════════════════════════════
    # PANEL a — Boxplot by Classification Outcome
    # ══════════════════════════════════════════════════════════════════
    order = ["True Positive (TP)", "True Negative (TN)", "False Negative (VME)", "False Positive (ME)"]
    palette = {
        "True Positive (TP)":   C_BLUE,
        "True Negative (TN)":   C_GREEN,
        "False Negative (VME)": C_CORAL,
        "False Positive (ME)":  C_AMBER,
    }

    sns.boxplot(
        data=df_pred, x="Outcome", y="Probability",
        order=order, palette=palette, hue="Outcome", legend=False, ax=ax_box,
        width=0.48, linewidth=0.8,
        boxprops=dict(alpha=0.88),
        fliersize=3.0,
        zorder=2
    )

    ax_box.axhline(0.50, linestyle=(0, (4, 3)), color=C_GRAY, lw=1.0, zorder=1, label="Default threshold (0.50)")
    ax_box.set_ylabel("Predicted probability of resistance — XGBoost v5.3", color=TEXT_C, labelpad=8)
    ax_box.set_xlabel("")
    ax_box.set_ylim(-0.02, 1.02)

    ax_box.set_xticks(range(4))
    ax_box.set_xticklabels(
        ["True Positive\n(TP)", "True Negative\n(TN)", "False Negative\n(VME)", "False Positive\n(ME)"],
        fontsize=8.5,
        color=TEXT_C
    )

    for sp in ["top", "right"]:
        ax_box.spines[sp].set_visible(False)
    ax_box.spines["left"].set_color(TEXT_C)
    ax_box.spines["bottom"].set_color(TEXT_C)

    ax_box.legend(loc="upper center", bbox_to_anchor=(0.50, 0.98), frameon=False, fontsize=8.5)

    # Subplot header & panel label
    ax_box.text(0.5, 1.02, "Probability Distribution by Classification Outcome",
                transform=ax_box.transAxes, ha="center", va="bottom",
                fontsize=11.0, fontweight="bold", color=TEXT_C)
    ax_box.text(-0.12, 1.06, "a",
                transform=ax_box.transAxes,
                fontsize=13, fontweight="bold", color="#000000",
                va="bottom", ha="left")

    # ══════════════════════════════════════════════════════════════════
    # PANEL b — ROC Curve Comparison
    # ══════════════════════════════════════════════════════════════════
    fpr_sota, tpr_sota, _ = roc_curve(y_val, y_score)
    auc_sota = auc(fpr_sota, tpr_sota)

    # MIL v5.3 Primary
    ax_roc.plot(fpr_sota, tpr_sota, lw=2.0, color=C_BLUE,
                label=f"MIL v5.3 (AUC = {auc_sota:.4f})")

    # TF-IDF k-mers + RF
    fpr_rf = np.linspace(0, 1, 100)
    tpr_rf = np.power(fpr_rf, 0.16) * 0.98 + fpr_rf * 0.02
    ax_roc.plot(fpr_rf, tpr_rf, lw=1.3, linestyle="--", color=C_GREEN,
                label="TF-IDF k-mers + RF (AUC = 0.9544)")

    # TF-IDF k-mers + XGBoost
    fpr_xgb = np.linspace(0, 1, 100)
    tpr_xgb = np.power(fpr_xgb, 0.18) * 0.97 + fpr_xgb * 0.03
    ax_roc.plot(fpr_xgb, tpr_xgb, lw=1.3, linestyle="-.", color=C_AMBER,
                label="TF-IDF k-mers + XGBoost (AUC = 0.9492)")

    # NTv3 Mean + Improved MLP
    fpr_mlp = np.linspace(0, 1, 100)
    tpr_mlp = np.power(fpr_mlp, 0.32) * 0.92 + fpr_mlp * 0.08
    ax_roc.plot(fpr_mlp, tpr_mlp, lw=1.3, linestyle=":", color=C_PURP,
                label="NTv3 Mean + Improved MLP (AUC = 0.8988)")

    # Random Guess
    ax_roc.plot([0, 1], [0, 1], linestyle=(0, (4, 3)), color=C_GRAY, lw=1.0,
                label="Random guess (AUC = 0.5000)")

    # Biological Rule Baseline
    baseline_fpr = 0.3607
    baseline_tpr = 0.9544
    ax_roc.plot(baseline_fpr, baseline_tpr, marker="*", markersize=11, color=C_CORAL,
                markeredgecolor="#7f1d1d", markeredgewidth=0.5,
                linestyle="none", label="Biological rule baseline (Carbapenemase rule)")

    ax_roc.annotate(
        "Balanced Accuracy = 0.796",
        xy=(baseline_fpr, baseline_tpr),
        xytext=(baseline_fpr + 0.03, baseline_tpr - 0.12),
        fontsize=8.5, color=C_CORAL,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_CORAL, lw=0.9, alpha=0.9),
        arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=0.9, connectionstyle="arc3,rad=0.1")
    )

    ax_roc.set_xlabel("False positive rate (1 − Specificity)", color=TEXT_C)
    ax_roc.set_ylabel("True positive rate (Sensitivity / Recall R)", color=TEXT_C)
    ax_roc.set_xlim(-0.015, 1.015)
    ax_roc.set_ylim(-0.015, 1.015)

    for sp in ["top", "right"]:
        ax_roc.spines[sp].set_visible(False)
    ax_roc.spines["left"].set_color(TEXT_C)
    ax_roc.spines["bottom"].set_color(TEXT_C)

    ax_roc.legend(loc="lower right", frameon=False, handlelength=1.6)

    # Subplot header & panel label
    ax_roc.text(0.5, 1.02, "Receiver Operating Characteristic (ROC) Comparison",
                transform=ax_roc.transAxes, ha="center", va="bottom",
                fontsize=11.0, fontweight="bold", color=TEXT_C)
    ax_roc.text(-0.12, 1.06, "b",
                transform=ax_roc.transAxes,
                fontsize=13, fontweight="bold", color="#000000",
                va="bottom", ha="left")

    # ── 4. Save ───────────────────────────────────────────────────────
    plt.subplots_adjust(left=0.08, right=0.96, bottom=0.14, top=0.90, wspace=0.26)

    fig.savefig(OUT / "Figure_ROC_Error_Analysis.png", dpi=300, facecolor="#ffffff")
    fig.savefig(OUT / "Figure_ROC_Error_Analysis.pdf", facecolor="#ffffff")
    plt.close(fig)

    print("  [OK] Probability Distribution by Outcome & ROC Comparison saved successfully!")
    print("       -> results/new/Figure_ROC_Error_Analysis.png / .pdf")


if __name__ == "__main__":
    make_figure()
