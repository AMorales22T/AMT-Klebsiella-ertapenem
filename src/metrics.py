"""
Statistical evaluation utilities for antimicrobial resistance prediction models.
Includes DeLong's test for ROC-AUC variance and covariance, Clopper-Pearson exact
confidence intervals, and cluster-based bootstrap hypothesis testing.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC-AUC and its asymptotic variance using DeLong's method.
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

    ranks = stats.rankdata(predictions)
    pos_ranks = ranks[ground_truth == 1]
    neg_ranks = ranks[ground_truth == 0]

    auc = (np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

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

    return float(auc), float(variance)


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
    return float(cov_pos + cov_neg)


def compare_two_aucs(ground_truth, predictions1, predictions2):
    """
    Compares two paired AUCs using DeLong's test.
    Returns (auc1, auc2, z_score, p_value).
    """
    auc1, var1 = delong_roc_variance(ground_truth, predictions1)
    auc2, var2 = delong_roc_variance(ground_truth, predictions2)
    cov = delong_roc_covariance(ground_truth, predictions1, predictions2)

    var_diff = var1 + var2 - 2 * cov
    if var_diff <= 0:
        return auc1, auc2, 0.0, 1.0

    z = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, float(z), float(p_value)


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05):
    """
    Computes exact Clopper-Pearson binomial confidence interval for proportions.
    """
    if n == 0:
        return 0.0, 1.0
    lower = float(stats.beta.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    upper = float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return lower, upper


def cluster_bootstrap_compare(df, y_true_col, y_pred1_col, y_pred2_col, cluster_col, n_bootstraps=2000, seed=42):
    """
    Cluster bootstrap test for paired comparison of two models grouped by clusters (e.g. ST).
    """
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)

    cluster_indices = {c: [] for c in clusters}
    for i, c in enumerate(df[cluster_col].values):
        cluster_indices[c].append(i)

    y_true_all = df[y_true_col].values
    y_p1_all = df[y_pred1_col].values
    y_p2_all = df[y_pred2_col].values

    delta_aucs = []
    for _ in range(n_bootstraps):
        resampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        idx = [i for c in resampled_clusters for i in cluster_indices[c]]

        y_true = y_true_all[idx]
        if len(np.unique(y_true)) < 2:
            continue

        auc1 = roc_auc_score(y_true, y_p1_all[idx])
        auc2 = roc_auc_score(y_true, y_p2_all[idx])
        delta_aucs.append(auc1 - auc2)

    delta_aucs = np.array(delta_aucs)
    p_value = float(min(np.mean(delta_aucs <= 0) * 2, 1.0))
    ci_lower = float(np.percentile(delta_aucs, 2.5))
    ci_upper = float(np.percentile(delta_aucs, 97.5))

    return float(np.mean(delta_aucs)), ci_lower, ci_upper, p_value
