"""
Figure — Random-split vs. ST-blocked cross-validation AUC comparison.

Statistically rigorous version:
  Panel A: Random-split ROC-AUC (single run, flagged as n=1)
  Panel B: ST-blocked 5-fold CV AUC per fold (mean ± std)
  Panel C: ΔAUC with bootstrap CI from fold-level variation

Caption explicitly addresses clonal leakage bias.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

# ── Output ───────────────────────────────────────────────────────────
OUT = Path(__file__).resolve().parent.parent / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# DATA — edit this section when multi-run results become available
# ══════════════════════════════════════════════════════════════════════

# Random split: single 80/20 run (n=1).
# TODO: Replace with array of AUCs from repeated random splits
#       e.g. RANDOM_SPLIT_AUCS = [0.9621, 0.9587, 0.9644, 0.9610, 0.9598]
RANDOM_SPLIT_AUCS = [0.9621]  # single run → no std computable

# ST-blocked 5-fold CV
FOLD_AUCS = {
    "Fold 1": 0.8684,
    "Fold 2": 0.7869,
    "Fold 3": 0.7742,
    "Fold 4": 0.8548,
    "Fold 5": 0.7453,
}

# ── Derived statistics ───────────────────────────────────────────────
rs_arr      = np.array(RANDOM_SPLIT_AUCS)
RS_MEAN     = rs_arr.mean()
RS_STD      = rs_arr.std(ddof=1) if len(rs_arr) > 1 else None
RS_N        = len(rs_arr)

st_arr      = np.array(list(FOLD_AUCS.values()))
ST_MEAN     = st_arr.mean()
ST_STD      = st_arr.std(ddof=1)
ST_N        = len(st_arr)
ST_SEM      = ST_STD / np.sqrt(ST_N)
ST_CI95     = scipy_stats.t.interval(0.95, df=ST_N - 1,
                                      loc=ST_MEAN, scale=ST_SEM)

DELTA_AUC   = RS_MEAN - ST_MEAN

# Bootstrap CI for ΔAUC using fold-level resampling
rng = np.random.default_rng(42)
n_boot = 10_000
boot_deltas = []
for _ in range(n_boot):
    boot_st = rng.choice(st_arr, size=ST_N, replace=True)
    boot_rs = rng.choice(rs_arr, size=max(RS_N, ST_N), replace=True)
    boot_deltas.append(boot_rs.mean() - boot_st.mean())
boot_deltas = np.array(boot_deltas)
DELTA_CI_LO = np.percentile(boot_deltas, 2.5)
DELTA_CI_HI = np.percentile(boot_deltas, 97.5)

# One-sample t-test: is ΔAUC > 0? (H0: ST folds have same mean as RS)
# Under the null, each fold AUC = RS_MEAN. We test if observed fold means
# are significantly lower.
t_stat, p_value_onesided = scipy_stats.ttest_1samp(st_arr, RS_MEAN)
p_value_onesided = p_value_onesided / 2  # one-sided (ST < RS)

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

# Palette
COL_RANDOM   = "#2563EB"
COL_ST       = "#0891B2"
COL_ST_BARS  = "#22D3EE"
COL_MEAN     = "#DC2626"
COL_DELTA    = "#F97316"
COL_DROP_BAR = "#EF4444"
BG           = "#FFFFFF"
TEXT         = "#1E293B"
MUTED        = "#64748B"
WARN         = "#B45309"   # amber for n=1 warning


def make_figure():
    fig = plt.figure(figsize=(12.5, 5.2), facecolor=BG)

    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.8, 1.3],
        left=0.07, right=0.97, bottom=0.22, top=0.83,
        wspace=0.35,
    )

    ax_b = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[1])

    # ══════════════════════════════════════════════════════════════════
    # LEFT PANEL — ST-blocked 5-fold CV per fold
    # ══════════════════════════════════════════════════════════════════

    fold_names = list(FOLD_AUCS.keys())
    fold_vals  = list(FOLD_AUCS.values())
    x_pos      = np.arange(len(fold_names))

    bars = ax_b.bar(x_pos, fold_vals, width=0.58, color=COL_ST_BARS,
                    edgecolor=COL_ST, linewidth=1.0, zorder=3, alpha=0.85)

    # Mean ± std band
    ax_b.axhspan(ST_MEAN - ST_STD, ST_MEAN + ST_STD,
                 color=COL_MEAN, alpha=0.06, zorder=1)
    ax_b.axhline(ST_MEAN, color=COL_MEAN, ls="--", lw=1.5, zorder=4,
                 label=f"Mean = {ST_MEAN:.4f} ± {ST_STD:.4f} (n = {ST_N})")

    # Random-split reference
    ax_b.axhline(RS_MEAN, color=COL_RANDOM, ls=":", lw=1.3, zorder=4,
                 alpha=0.7, label=f"Random split = {RS_MEAN:.4f} (n = {RS_N})")

    # Value annotations
    for bar, val in zip(bars, fold_vals):
        ax_b.text(bar.get_x() + bar.get_width() / 2, val + 0.010,
                  f"{val:.4f}", ha="center", va="bottom",
                  fontsize=8.5, fontweight="medium", color=TEXT, zorder=5)

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(fold_names, fontweight="medium")
    ax_b.set_ylabel("ROC-AUC")
    ax_b.set_ylim(0.65, 1.02)
    ax_b.set_title("ST-Blocked 5-Fold Cross-Validation",
                    fontweight="bold", pad=8)
    ax_b.legend(loc="upper right", framealpha=0.9, edgecolor="#E2E8F0")
    ax_b.grid(axis="y", alpha=0.25, zorder=0)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # ══════════════════════════════════════════════════════════════════
    # RIGHT PANEL — ΔAUC with CI and p-value
    # ══════════════════════════════════════════════════════════════════

    bar_width = 0.50

    # Random split bar
    ax_c.bar(0, RS_MEAN, width=bar_width,
             color=COL_RANDOM, alpha=0.18, edgecolor=COL_RANDOM,
             linewidth=1.2, zorder=3)
    if RS_STD is not None:
        ax_c.errorbar(0, RS_MEAN, yerr=RS_STD, fmt="none",
                      ecolor=COL_RANDOM, capsize=5, capthick=1.2,
                      elinewidth=1.2, zorder=5)

    # ST-blocked bar
    ax_c.bar(1, ST_MEAN, width=bar_width,
             color=COL_ST, alpha=0.18, edgecolor=COL_ST,
             linewidth=1.2, zorder=3)
    ax_c.errorbar(1, ST_MEAN, yerr=ST_STD, fmt="none",
                  ecolor=TEXT, capsize=5, capthick=1.2,
                  elinewidth=1.2, zorder=5)

    # Drop arrow
    arrow_x = 0.5
    ax_c.annotate(
        "", xy=(arrow_x, ST_MEAN + 0.005),
        xytext=(arrow_x, RS_MEAN - 0.005),
        arrowprops=dict(arrowstyle="-|>", color=COL_DROP_BAR,
                        lw=2.5, mutation_scale=18),
        zorder=6,
    )

    # ── ΔAUC label with bootstrap CI ─────────────────────────────────
    pct_drop = (DELTA_AUC / RS_MEAN) * 100

    label_x = 1.50
    mid_y   = (RS_MEAN + ST_MEAN) / 2

    ax_c.text(label_x, mid_y + 0.045,
              f"ΔAUC = −{DELTA_AUC:.4f}",
              ha="left", va="center", fontsize=10.5, fontweight="bold",
              color=COL_DROP_BAR, zorder=6)

    ax_c.text(label_x, mid_y + 0.01,
              f"95% CI [{DELTA_CI_LO:.3f}, {DELTA_CI_HI:.3f}]",
              ha="left", va="center", fontsize=8, color=MUTED, zorder=6)

    ax_c.text(label_x, mid_y - 0.020,
              f"({pct_drop:.1f}% drop)",
              ha="left", va="center", fontsize=8.5, color=MUTED,
              fontstyle="italic", zorder=6)

    # p-value annotation
    if p_value_onesided < 0.001:
        p_str = "p < 0.001"
    elif p_value_onesided < 0.01:
        p_str = f"p = {p_value_onesided:.3f}"
    else:
        p_str = f"p = {p_value_onesided:.3f}"

    ax_c.text(label_x, mid_y - 0.055,
              f"{p_str} (one-sided t-test†)",
              ha="left", va="center", fontsize=7.5, fontweight="bold",
              color=TEXT, zorder=6)

    # Value labels on bars
    # Random split
    rs_label = f"{RS_MEAN:.4f}"
    if RS_STD is not None:
        rs_label += f" ± {RS_STD:.4f}"
    else:
        rs_label += "  (n=1)"
    ax_c.text(0, RS_MEAN + 0.015, rs_label,
              ha="center", va="bottom", fontsize=8.5, fontweight="bold",
              color=COL_RANDOM, zorder=5)

    ax_c.text(1, ST_MEAN + ST_STD + 0.015,
              f"{ST_MEAN:.4f} ± {ST_STD:.4f}",
              ha="center", va="bottom", fontsize=8.5, fontweight="bold",
              color=COL_ST, zorder=5)

    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["Random\nSplit", "ST-Blocked\nCV (mean)"],
                          fontweight="medium", fontsize=9)
    ax_c.set_ylabel("ROC-AUC")
    ax_c.set_ylim(0.65, 1.08)
    ax_c.set_xlim(-0.55, 2.85)
    ax_c.set_title("Performance Loss Under\nST-Controlled Evaluation",
                    fontweight="bold", pad=6, fontsize=10.5)
    ax_c.grid(axis="y", alpha=0.25, zorder=0)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # ══════════════════════════════════════════════════════════════════
    # SUPTITLE
    # ══════════════════════════════════════════════════════════════════
    fig.suptitle(
        "Clonal Leakage Effect on Model Performance — "
        "MIL v5.3 + XGBoost  |  K. pneumoniae / Ertapenem",
        fontsize=12, fontweight="bold", color=TEXT, y=0.96,
    )

    # ══════════════════════════════════════════════════════════════════
    # CAPTION (below the figure)
    # ══════════════════════════════════════════════════════════════════
    caption = (
        "Figure. Random-split evaluation inflates ROC-AUC due to clonal structure in the dataset. "
        "Isolates sharing the same sequence type (ST) are distributed across training and test sets "
        "under random splitting, allowing the model to exploit clonal similarity rather than learning "
        "true resistance determinants. "
        f"ST-blocked 5-fold CV (mean AUC = {ST_MEAN:.4f} ± {ST_STD:.4f}) "
        f"reveals a {pct_drop:.1f}% performance drop ({p_str}) "
        "relative to the random split, quantifying the magnitude of clonal leakage bias. "
        "†One-sided t-test of ST-blocked fold AUCs vs. the random-split point estimate; "
        "a paired DeLong test requires per-sample predicted probabilities from both protocols."
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom",
             fontsize=7.5, color=MUTED, wrap=True,
             transform=fig.transFigure,
             fontstyle="italic",
             multialignment="left",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC",
                       edgecolor="#E2E8F0", linewidth=0.5))

    # ══════════════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════════════
    for fmt in ("png", "pdf"):
        path = OUT / f"Figure_AUC_Comparison.{fmt}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.15,
                    facecolor=BG, dpi=300)
        print(f"  [OK] {path.name}")
    plt.close(fig)

    # Print summary statistics
    print(f"\n  -- Summary ------------------------------------")
    print(f"  Random split:  {RS_MEAN:.4f}  (n = {RS_N})")
    if RS_STD is not None:
        print(f"                 ± {RS_STD:.4f}")
    print(f"  ST-blocked:    {ST_MEAN:.4f} +/- {ST_STD:.4f}  (n = {ST_N})")
    print(f"  95% CI ST:     [{ST_CI95[0]:.4f}, {ST_CI95[1]:.4f}]")
    print(f"  dAUC:          -{DELTA_AUC:.4f}  ({pct_drop:.1f}% drop)")
    print(f"  dAUC 95% CI:   [{DELTA_CI_LO:.4f}, {DELTA_CI_HI:.4f}]")
    print(f"  t-stat:        {t_stat:.3f}")
    print(f"  p-value (1-s): {p_value_onesided:.6f}")
    print(f"  --------------------------------------------")


if __name__ == "__main__":
    make_figure()
