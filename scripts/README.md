# Analysis & Reproduction Scripts (`scripts/`)

All scripts in this directory are standalone and use relative paths to reproduce tables, statistical tests, and figures directly from the repository data.

## Execution

Ensure dependencies are installed (`pip install -r requirements.txt`), then run:

### 1. Primary Statistical Metrics (`compute_stats.py`)
```bash
python scripts/compute_stats.py
```
- Computes ROC-AUC and DeLong 95% confidence intervals for:
  - ST-Blocked CV (MIL+XGBoost): `AUC = 0.7981 (95% CI: [0.7837, 0.8125])`
  - ST-Blocked CV (MIL-only): `AUC = 0.8111 (95% CI: [0.7972, 0.8250])`
  - Random Split (with leakage): `AUC = 0.9621 (95% CI: [0.9500, 0.9743])`
- Computes paired DeLong test between MIL+XGBoost vs MIL-only (ΔAUC = -0.0130, p = 0.0059).
- Computes confusion matrix, sensitivity, specificity, VME (53.59%), and ME (12.29%) with Clopper-Pearson exact binomial confidence intervals.

### 2. Main Clonal Leakage Figure (`figure_auc_comparison.py`)
```bash
python scripts/figure_auc_comparison.py
```
- Reproduces `results/figures/Figure_AUC_Comparison.png` and `.pdf` comparing Random Split vs. per-fold ST-blocked CV with fold-level bootstrap CI and one-sided t-test (p = 0.0014).

### 3. ROC Curves & Baseline Comparison (`plot_paper_figure_options.py`)
```bash
python scripts/plot_paper_figure_options.py
```
- Generates `results/figures/Figure_AUC_OptionA_Subset.png` (evaluating against the biological rule baseline on the 487-isolate subset) and `Figure_AUC_OptionB_Full.png` (full OOF vector).

### 4. McNemar Statistical Test (`run_mcnemar.py`)
```bash
python scripts/run_mcnemar.py
```
- Builds 2x2 contingency tables comparing the Machine Learning predictions vs. Carbapenemase biological rule baseline and evaluates statistical significance with exact binomial McNemar test.

### 5. Cluster Bootstrap Hypothesis Test (`test_bootstrap.py`)
```bash
python scripts/test_bootstrap.py
```
- Performs cluster-based bootstrap resampling across the 405 Sequence Type (ST) clusters (B = 1,000 iterations) to assess statistical significance while accounting for clonal correlation.
