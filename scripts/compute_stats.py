import numpy as np
import pandas as pd
from scipy import stats

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
    
    # number of positive and negative samples
    n_pos = np.sum(ground_truth == 1)
    n_neg = np.sum(ground_truth == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0
    
    # Wilcoxon-Mann-Whitney U-statistic
    ranks = stats.rankdata(predictions)
    pos_ranks = ranks[ground_truth == 1]
    neg_ranks = ranks[ground_truth == 0]
    
    # AUC
    auc = (np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    
    # DeLong components
    # For positive samples
    # number of negatives with score less than each positive
    v_pos = np.zeros(n_pos)
    # For negative samples
    # number of positives with score greater than each negative
    v_neg = np.zeros(n_neg)
    
    pred_pos = predictions[ground_truth == 1]
    pred_neg = predictions[ground_truth == 0]
    
    # Fast vectorized calculation
    for i, p in enumerate(pred_pos):
        v_pos[i] = (np.sum(pred_neg < p) + 0.5 * np.sum(pred_neg == p)) / n_neg
        
    for i, n in enumerate(pred_neg):
        v_neg[i] = (np.sum(pred_pos > n) + 0.5 * np.sum(pred_pos == n)) / n_pos
        
    # Variance
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
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, z, p_value

def clopper_pearson_ci(k, n, alpha=0.05):
    """
    Computes Clopper-Pearson exact binomial confidence interval.
    """
    if n == 0:
        return 0.0, 1.0
    lower = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    upper = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lower, upper

# Load data
from pathlib import Path
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
print(f"ST-Blocked CV (MIL+XGBoost) AUC: {auc_oof:.4f} (95% CI: [{ci_oof[0]:.4f}, {ci_oof[1]:.4f}])")
print(f"ST-Blocked CV (MIL-only) AUC:    {auc_mil:.4f} (95% CI: [{ci_mil[0]:.4f}, {ci_mil[1]:.4f}])")
print(f"Random Split AUC:                {auc_rs:.4f} (95% CI: [{ci_rs[0]:.4f}, {ci_rs[1]:.4f}])")
print()
print(f"Comparison (MIL+XGBoost vs MIL-only):")
print(f"  Delta AUC:  {auc1 - auc2:+.4f}")
print(f"  z-statistic: {z_stat:.4f}")
print(f"  p-value:     {p_val:.6f}")

# --- 3. Compute optimal threshold and confusion matrix metrics ---
# We'll use 0.5 as default threshold
y_true = oof['y_true'].values
y_pred = (oof['y_score'].values >= 0.5).astype(int)

tp = np.sum((y_true == 1) & (y_pred == 1))
tn = np.sum((y_true == 0) & (y_pred == 0))
fp = np.sum((y_true == 0) & (y_pred == 1))
fn = np.sum((y_true == 1) & (y_pred == 0))

n_r = np.sum(y_true == 1)
n_s = np.sum(y_true == 0)

sens = tp / n_r
spec = tn / n_s
vme = fn / n_r  # Very Major Error (false susceptible rate)
me = fp / n_s   # Major Error (false resistant rate)

ci_sens = clopper_pearson_ci(tp, n_r)
ci_spec = clopper_pearson_ci(tn, n_s)
ci_vme = clopper_pearson_ci(fn, n_r)
ci_me = clopper_pearson_ci(fp, n_s)

print()
print("=== CONFUSION MATRIX & ERRORS (Threshold = 0.5) ===")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Sensitivity (TPR):         {sens:.4f} (95% CI: [{ci_sens[0]:.4f}, {ci_sens[1]:.4f}])")
print(f"Specificity (TNR):         {spec:.4f} (95% CI: [{ci_spec[0]:.4f}, {ci_spec[1]:.4f}])")
print(f"Very Major Error (VME):    {vme:.4f} (95% CI: [{ci_vme[0]:.4f}, {ci_vme[1]:.4f}])")
print(f"Major Error (ME):          {me:.4f} (95% CI: [{ci_me[0]:.4f}, {ci_me[1]:.4f}])")
