import numpy as np
import pandas as pd
import math
from pathlib import Path

try:
    from scipy import stats
    has_scipy = True
except ImportError:
    has_scipy = False

def rankdata(a):
    if has_scipy:
        return stats.rankdata(a)
    a = np.asarray(a)
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    i = 0
    while i < len(a):
        j = i
        while j < len(a) - 1 and a[order[j]] == a[order[j+1]]:
            j += 1
        if j > i:
            avg_rank = (i + 1 + j + 1) / 2.0
            ranks[order[i:j+1]] = avg_rank
            i = j + 1
        else:
            i += 1
    return ranks

def norm_cdf(x):
    if has_scipy:
        return stats.norm.cdf(x)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC-AUC variance using DeLong's method.
    Adapted from: https://github.com/yandexdataschool/roc_comparison
    """
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    
    order = np.argsort(predictions)
    ground_truth = ground_truth[order]
    predictions = predictions[order]
    
    n_pos = np.sum(ground_truth == 1)
    n_neg = np.sum(ground_truth == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0
    
    pos_ranks = rankdata(predictions)[ground_truth == 1]
    auc = (np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    
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
    variance = var_pos + var_neg
    
    return auc, variance

def delong_roc_covariance(ground_truth, predictions1, predictions2):
    """
    Computes covariance between two ROC-AUCs using DeLong's method.
    """
    ground_truth = np.array(ground_truth)
    predictions1 = np.array(predictions1)
    predictions2 = np.array(predictions2)
    
    n_pos = np.sum(ground_truth == 1)
    n_neg = np.sum(ground_truth == 0)
    
    pred_pos1 = predictions1[ground_truth == 1]
    pred_neg1 = predictions1[ground_truth == 0]
    pred_pos2 = predictions2[ground_truth == 1]
    pred_neg2 = predictions2[ground_truth == 0]
    
    v_pos1 = np.zeros(n_pos)
    v_neg1 = np.zeros(n_neg)
    v_pos2 = np.zeros(n_pos)
    v_neg2 = np.zeros(n_neg)
    
    for i in range(n_pos):
        v_pos1[i] = (np.sum(pred_neg1 < pred_pos1[i]) + 0.5 * np.sum(pred_neg1 == pred_pos1[i])) / n_neg
        v_pos2[i] = (np.sum(pred_neg2 < pred_pos2[i]) + 0.5 * np.sum(pred_neg2 == pred_pos2[i])) / n_neg
        
    for i in range(n_neg):
        v_neg1[i] = (np.sum(pred_pos1 > pred_neg1[i]) + 0.5 * np.sum(pred_pos1 == pred_neg1[i])) / n_pos
        v_neg2[i] = (np.sum(pred_pos2 > pred_neg2[i]) + 0.5 * np.sum(pred_pos2 == pred_neg2[i])) / n_pos
        
    cov_pos = np.cov(v_pos1, v_pos2, ddof=1)[0, 1] / n_pos
    cov_neg = np.cov(v_neg1, v_neg2, ddof=1)[0, 1] / n_neg
    covariance = cov_pos + cov_neg
    
    return covariance

def compare_two_aucs(ground_truth, predictions1, predictions2):
    """
    Compares two AUCs using DeLong's test. Returns z-score and two-sided p-value.
    """
    auc1, var1 = delong_roc_variance(ground_truth, predictions1)
    auc2, var2 = delong_roc_variance(ground_truth, predictions2)
    cov = delong_roc_covariance(ground_truth, predictions1, predictions2)
    
    var_diff = var1 + var2 - 2 * cov
    if var_diff <= 0:
        return auc1, auc2, 0.0, 1.0
    
    z = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * (1 - norm_cdf(abs(z)))
    return auc1, auc2, z, p_value

def clopper_pearson_ci(k, n, alpha=0.05):
    """
    Computes Clopper-Pearson exact binomial confidence interval.
    """
    if n == 0:
        return 0.0, 1.0
    if has_scipy:
        lower = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
        upper = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
        return lower, upper
    # Wilson score interval as graceful fallback when scipy is not available
    p = k / n
    z = 1.95996
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

def compute_pr_auc(y_true, y_scores):
    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_scores)
    y_sort = y_true[order]
    tps = np.cumsum(y_sort == 1)
    fps = np.cumsum(y_sort == 0)
    recalls = tps / n_pos
    precisions = tps / (tps + fps)
    return float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))

# Load data
BASE_DIR = Path(__file__).resolve().parent.parent
oof = pd.read_csv(BASE_DIR / 'results' / 'oof_predictions_pooled.csv')
rs = pd.read_csv(BASE_DIR / 'results' / 'random_split_predictions.csv')

# --- 1. Compute AUCs and CIs ---
auc_oof, var_oof = delong_roc_variance(oof['y_true'], oof['y_score'])
ci_oof = (auc_oof - 1.96 * np.sqrt(var_oof), auc_oof + 1.96 * np.sqrt(var_oof))

auc_mil, var_mil = delong_roc_variance(oof['y_true'], oof['y_score_mil'])
ci_mil = (auc_mil - 1.96 * np.sqrt(var_mil), auc_mil + 1.96 * np.sqrt(var_mil))

auc_rs, var_rs = delong_roc_variance(rs['y_true'], rs['y_score'])
ci_rs = (auc_rs - 1.96 * np.sqrt(var_rs), auc_rs + 1.96 * np.sqrt(var_rs))

# --- 2. Compare MIL+XGBoost vs MIL-only (ablation) ---
auc1, auc2, z_stat, p_val = compare_two_aucs(oof['y_true'], oof['y_score'], oof['y_score_mil'])

print("=== STATISTICAL ANALYSIS RESULTS ===")
print(f"ST-Blocked CV (MIL-only) AUC:    {auc_mil:.4f} (95% CI: [{ci_mil[0]:.4f}, {ci_mil[1]:.4f}])")
print(f"ST-Blocked CV (MIL+XGBoost) AUC: {auc_oof:.4f} (95% CI: [{ci_oof[0]:.4f}, {ci_oof[1]:.4f}])")
print(f"Random Split AUC:                {auc_rs:.4f} (95% CI: [{ci_rs[0]:.4f}, {ci_rs[1]:.4f}])")
print()
print(f"Comparison (MIL+XGBoost vs MIL-only under ST-blocking):")
print(f"  Delta AUC:   {auc1 - auc2:+.4f}")
print(f"  z-statistic: {z_stat:.4f}")
print(f"  p-value:     {p_val:.6f}")

# --- 3. Precision-Recall AUC (PR-AUC) ---
pr_auc_mil = compute_pr_auc(oof['y_true'].values, oof['y_score_mil'].values)
pr_auc_xgb = compute_pr_auc(oof['y_true'].values, oof['y_score'].values)
pr_auc_rs = compute_pr_auc(rs['y_true'].values, rs['y_score'].values)

print("\n=== PRECISION-RECALL AUC (PR-AUC) ===")
print(f"ST-Blocked CV (MIL-only) PR-AUC:    {pr_auc_mil:.4f} (baseline prevalence: {np.mean(oof['y_true']):.1%})")
print(f"ST-Blocked CV (MIL+XGB) PR-AUC:     {pr_auc_xgb:.4f}")
print(f"Random Split (Leakage) PR-AUC:      {pr_auc_rs:.4f} (baseline prevalence: {np.mean(rs['y_true']):.1%})")

# --- 4. Confusion Matrix & Errors at Default Threshold 0.50 ---
n_r = np.sum(oof['y_true'] == 1)
n_s = np.sum(oof['y_true'] == 0)

for model_name, score_col in [("MIL-only (Primary)", "y_score_mil"), ("MIL+XGBoost (Ablation)", "y_score")]:
    y_pred = (oof[score_col].values >= 0.5).astype(int)
    tp = np.sum((oof['y_true'] == 1) & (y_pred == 1))
    tn = np.sum((oof['y_true'] == 0) & (y_pred == 0))
    fp = np.sum((oof['y_true'] == 0) & (y_pred == 1))
    fn = np.sum((oof['y_true'] == 1) & (y_pred == 0))
    
    sens = tp / n_r
    spec = tn / n_s
    vme = fn / n_r
    me = fp / n_s
    
    ci_vme = clopper_pearson_ci(fn, n_r)
    ci_me = clopper_pearson_ci(fp, n_s)
    
    print(f"\n=== {model_name} Error Rates (Threshold = 0.50, n=3,702) ===")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Sensitivity (Recall R):    {sens:.4f}")
    print(f"Specificity (TNR):         {spec:.4f}")
    print(f"Very Major Error (VME):    {vme:.4f} (95% CI: [{ci_vme[0]:.4f}, {ci_vme[1]:.4f}])")
    print(f"Major Error (ME):          {me:.4f} (95% CI: [{ci_me[0]:.4f}, {ci_me[1]:.4f}])")

# --- 5. ST-Blocked Threshold Sweep for MIL-only ---
print("\n=== ST-BLOCKED THRESHOLD SWEEP (MIL-only, n=3,702) ===")
print("Thresh\tSensitivity\tSpecificity\tVME (fn/R)\tME (fp/S)\tF1")
for th in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
    p = (oof['y_score_mil'].values >= th).astype(int)
    _tp = np.sum((oof['y_true'] == 1) & (p == 1))
    _fn = np.sum((oof['y_true'] == 1) & (p == 0))
    _fp = np.sum((oof['y_true'] == 0) & (p == 1))
    _tn = np.sum((oof['y_true'] == 0) & (p == 0))
    _sens = _tp / n_r
    _spec = _tn / n_s
    _prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0
    _f1 = 2 * _prec * _sens / (_prec + _sens) if (_prec + _sens) > 0 else 0
    print(f"{th:.2f}\t{_sens:.1%}\t\t{_spec:.1%}\t\t{_fn/n_r:.1%}\t\t{_fp/n_s:.1%}\t\t{_f1:.4f}")
