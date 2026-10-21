# Analysis & Reproduction Scripts (`scripts/`)

All scripts in this directory are standalone and use relative paths to reproduce tables, statistical tests, and figures directly from the repository data.

Figure scripts are cleanly partitioned into **`scripts/new/`** (active Nature Portfolio main figures), **`scripts/supplementary/`** (supplementary figures suite), and **`scripts/legacy/`** (historical reference code).

---

## Directory Organization

```
scripts/
├── new/                             # Current Nature Portfolio publication figures (Figures 1–6)
│   ├── refine_figure1.py            # Generates Figure 1 (Framework & resistome architecture)
│   ├── plot_paper_figure2.py        # Generates Figure 2 (ROC curves & DeLong validation bars)
│   ├── plot_roc_error_analysis.py   # Generates Outcome probability distribution & ROC comparison
│   ├── plot_paper_figure3.py        # Generates Figure 3 (SHAP global beeswarm attribution)
│   ├── plot_paper_figure4_ig.py     # Generates Figure 4 (blaKPC-2 Integrated Gradients profile)
│   ├── plot_paper_figure5_logo.py   # Generates Figure 5 (LOGO sensitivity & top 20 driver genes)
│   └── plot_paper_figure6_pr_calibration.py # Generates Figure 6 (PR curve & calibration diagram)
├── supplementary/                   # Supplementary publication figures suite (Figures S1–S5)
│   └── plot_supplementary_figures.py# Generates Supplementary Figures S1 through S5
├── legacy/                          # Historical reference scripts
│   ├── figure_auc_comparison.py     # Original clonal leakage comparison script
│   ├── plot_paper_figure_options.py # Original draft figure options (Option A & B)
│   ├── generate_figures.py          # Unified historical script
│   ├── generate_shap_activation_fig4.py
│   └── ig_kpc2_plots.py
├── compute_stats.py                 # Primary statistical metrics & Table 1 calculations
├── run_mcnemar.py                   # McNemar statistical hypothesis tests (ML vs. rule)
└── test_bootstrap.py                # ST-level cluster bootstrap resampling tests
```

---

## Execution Guide

### 1. Generating Nature Portfolio Publication Figures (`scripts/new/`)

```bash
# Figure 1: Architecture & biological resistome diagram
python scripts/new/refine_figure1.py

# Figure 2: ROC curves & clonal inflation drop
python scripts/new/plot_paper_figure2.py

# Validation Benchmark: Outcome probability distribution & ROC comparison
python scripts/new/plot_roc_error_analysis.py

# Figure 3: SHAP global feature attributions
python scripts/new/plot_paper_figure3.py

# Figure 4: Integrated Gradients blaKPC-2 CDS sequence attribution
python scripts/new/plot_paper_figure4_ig.py

# Figure 5: Leave-One-Gene-Out (LOGO) sensitivity & top 20 driver genes
python scripts/new/plot_paper_figure5_logo.py

# Figure 6: Precision–Recall curve & model calibration analysis
python scripts/new/plot_paper_figure6_pr_calibration.py
```
*Outputs are saved directly to `results/new/`.*

---

### 2. Generating Supplementary Figures (`scripts/supplementary/`)

```bash
# Supplementary Figures S1–S5: Threshold sweeps, attention, error profiles, cohort flow, UMAP
python scripts/supplementary/plot_supplementary_figures.py
```
*Outputs are saved directly to `results/supplementary/`.*

---

### 3. Primary Statistical Metrics & Hypothesis Tests

```bash
# Primary metrics (ROC-AUC, DeLong 95% CIs, VME, ME, paired tests)
python scripts/compute_stats.py

# McNemar 2x2 contingency table test (ML vs. biological carbapenemase rule)
python scripts/run_mcnemar.py

# Cluster bootstrap hypothesis test across 405 Sequence Types (B = 1,000 iterations)
python scripts/test_bootstrap.py
```

---

### 3. Historical / Legacy Figures (`scripts/legacy/`)

```bash
# Original clonal leakage comparison
python scripts/legacy/figure_auc_comparison.py
```
