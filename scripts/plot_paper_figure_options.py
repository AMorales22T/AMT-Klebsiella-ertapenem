import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import stats
import os

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
    neg_ranks = ranks[ground_truth == 0]
    
    auc_val = (np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    
    v_pos = np.zeros(n_pos)
    v_neg = np.zeros(n_neg)
    
    pred_pos = predictions[ground_truth == 1]
    pred_neg = predictions[ground_truth == 0]
    
    for i, p in enumerate(pred_pos):
        v_pos[i] = (np.sum(pred_neg < p) + 0.5 * np.sum(pred_neg == p)) / n_neg
        
    for i, n in enumerate(pred_neg):
        v_neg[i] = (np.sum(pred_pos > n) + 0.5 * np.sum(pred_pos == n)) / n_pos
        
    var_pos = np.var(v_pos, ddof=1) / n_pos
    var_neg = np.var(v_neg, ddof=1) / n_neg
    variance = var_pos + var_neg
    
    return auc_val, variance

def cluster_bootstrap_compare(df, y_true_col, y_pred1_col, y_pred2_col, cluster_col, n_bootstraps=2000):
    np.random.seed(42)
    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)
    
    # Pre-map data to clusters for fast lookup
    cluster_indices = {c: [] for c in clusters}
    cluster_vals = df[cluster_col].values
    for i, c in enumerate(cluster_vals):
        cluster_indices[c].append(i)
        
    y_true_all = df[y_true_col].values
    y_p1_all = df[y_pred1_col].values
    y_p2_all = df[y_pred2_col].values
    
    delta_aucs = []
    from sklearn.metrics import roc_auc_score
    for _ in range(n_bootstraps):
        resampled_clusters = np.random.choice(clusters, size=n_clusters, replace=True)
        
        # Flatten the list of lists
        idx = [i for c in resampled_clusters for i in cluster_indices[c]]
        
        y_true = y_true_all[idx]
        y_p1 = y_p1_all[idx]
        y_p2 = y_p2_all[idx]
        
        if len(np.unique(y_true)) < 2:
            continue
            
        auc1 = roc_auc_score(y_true, y_p1)
        auc2 = roc_auc_score(y_true, y_p2)
        delta_aucs.append(auc1 - auc2)
        
    delta_aucs = np.array(delta_aucs)
    p_value = np.mean(delta_aucs <= 0) * 2  # two-sided
    p_value = min(p_value, 1.0)
    ci_lower = np.percentile(delta_aucs, 2.5)
    ci_upper = np.percentile(delta_aucs, 97.5)
    
    return np.mean(delta_aucs), ci_lower, ci_upper, p_value

def generate_figure(oof, rs, option, out_path):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "figure.dpi": 300,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor='white')
    ax_roc, ax_bar = axes

    # 1. Random Split
    fpr_rs, tpr_rs, _ = roc_curve(rs['y_true'], rs['y_score'])
    auc_rs, var_rs = delong_roc_variance(rs['y_true'], rs['y_score'])
    ci_rs = 1.96 * np.sqrt(var_rs)

    # 2. ST-Blocked (MIL-only)
    fpr_mil, tpr_mil, _ = roc_curve(oof['y_true'], oof['y_score_mil'])
    auc_mil, var_mil = delong_roc_variance(oof['y_true'], oof['y_score_mil'])
    ci_mil = 1.96 * np.sqrt(var_mil)

    # 3. ST-Blocked (MIL+XGBoost)
    fpr_xgb, tpr_xgb, _ = roc_curve(oof['y_true'], oof['y_score'])
    auc_xgb, var_xgb = delong_roc_variance(oof['y_true'], oof['y_score'])
    ci_xgb = 1.96 * np.sqrt(var_xgb)

    # Plot ROCs
    ax_roc.plot(fpr_rs, tpr_rs, color='#2563EB', lw=2, label=f'Random Split\nAUC = {auc_rs:.4f} (95% CI: ±{ci_rs:.4f})')
    ax_roc.plot(fpr_mil, tpr_mil, color='#0D9488', lw=2, label=f'ST-Blocked CV (MIL-only)\nAUC = {auc_mil:.4f} (95% CI: ±{ci_mil:.4f})')
    ax_roc.plot(fpr_xgb, tpr_xgb, color='#7C3AED', lw=2, label=f'ST-Blocked CV (MIL+XGBoost)\nAUC = {auc_xgb:.4f} (95% CI: ±{ci_xgb:.4f})')
    
    if option == 'A':
        # Baseline on overlap
        ax_roc.plot(0.9953, 1.0000, marker='*', color='#EF4444', markersize=12, ls='', label='Biological Rule Baseline\n(Calculated on 487 subset)')
        ax_roc.set_title('ROC Curves (Evaluated on 487 overlap subset)', fontweight='bold', pad=10)
    else:
        ax_roc.set_title('ROC Curves Comparison', fontweight='bold', pad=10)

    ax_roc.plot([0, 1], [0, 1], color='#94A3B8', ls='--', lw=1)
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)')
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)')
    ax_roc.legend(loc='lower right', framealpha=0.9)
    ax_roc.grid(True, alpha=0.15)

    # Panel B: Performance Bar Chart
    models = ['Random Split\n(Optimistic / Leakage)', 'ST-Blocked CV\n(MIL-only)', 'ST-Blocked CV\n(MIL+XGBoost)']
    aucs = [auc_rs, auc_mil, auc_xgb]
    errors = [ci_rs, ci_mil, ci_xgb]
    colors = ['#2563EB', '#0D9488', '#7C3AED']

    bars = ax_bar.bar(models, aucs, yerr=errors, color=colors, edgecolor='none', alpha=0.85, width=0.5, capsize=6, error_kw={'ecolor': '#1E293B', 'lw': 1.5})

    for bar, val in zip(bars, aucs):
        ax_bar.text(bar.get_x() + bar.get_width()/2, val - 0.08, f'{val:.4f}', ha='center', va='top', color='white', fontweight='bold', fontsize=10)

    loss_pct_mil = ((auc_rs - auc_mil) / auc_rs) * 100
    ax_bar.annotate('', xy=(1, auc_mil + 0.02), xytext=(0, auc_rs - 0.02), arrowprops=dict(arrowstyle="->", color='#EF4444', lw=2.0, ls=':'))
    ax_bar.text(0.5, (auc_rs + auc_mil)/2 + 0.04, f'-{auc_rs - auc_mil:.4f} ({loss_pct_mil:.1f}% drop)\nClonal leakage effect', ha='center', va='bottom', color='#EF4444', fontsize=9, fontweight='semibold')

    ax_bar.set_ylabel('ROC-AUC')
    ax_bar.set_ylim([0.6, 1.02])
    ax_bar.set_title('Clonal Leakage Effect on Performance', fontweight='bold', pad=10)
    ax_bar.grid(axis='y', alpha=0.15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent
    RESULTS_DIR = BASE_DIR / "results"
    FIGURES_DIR = RESULTS_DIR / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(RESULTS_DIR / "oof_predictions_pooled.csv")
    rs = pd.read_csv(RESULTS_DIR / "random_split_predictions.csv")
    
    # Do cluster bootstrap for full OOF
    print("Running ST-level cluster bootstrap on full OOF...")
    mean_delta, ci_l, ci_u, pval = cluster_bootstrap_compare(oof, 'y_true', 'y_score', 'y_score_mil', 'ST', n_bootstraps=2000)
    print(f"MIL+XGBoost vs MIL-only (Full OOF):")
    print(f"Delta AUC = {mean_delta:.4f} (95% CI: [{ci_l:.4f}, {ci_u:.4f}]), p = {pval:.4f}")
    
    # Generate Option B
    fig_b_path = FIGURES_DIR / "Figure_AUC_OptionB_Full.png"
    generate_figure(oof, rs, 'B', fig_b_path)
    print(f"Saved: {fig_b_path}")
    
    # Generate Option A
    oof_subset = oof[oof['has_carbapenemase'].notna()].copy()
    fig_a_path = FIGURES_DIR / "Figure_AUC_OptionA_Subset.png"
    generate_figure(oof_subset, rs, 'A', fig_a_path)
    print(f"Saved: {fig_a_path}")
    
    # Run cluster bootstrap for Subset OOF just in case
    print("Running ST-level cluster bootstrap on Subset OOF...")
    mean_delta_sub, ci_l_sub, ci_u_sub, pval_sub = cluster_bootstrap_compare(oof_subset, 'y_true', 'y_score', 'y_score_mil', 'ST', n_bootstraps=2000)
    print(f"MIL+XGBoost vs MIL-only (Subset OOF):")
    print(f"Delta AUC = {mean_delta_sub:.4f} (95% CI: [{ci_l_sub:.4f}, {ci_u_sub:.4f}]), p = {pval_sub:.4f}")
    
    print("Both figure versions saved.")
