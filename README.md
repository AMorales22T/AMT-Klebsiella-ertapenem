# Lineage-aware genomic learning exposes clonal inflation in ertapenem resistance prediction

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22753637.svg)](https://doi.org/10.5281/zenodo.22753637)

This repository contains the complete reproducible codebase, analysis scripts, metadata manifests, and prediction outputs supporting the study on genomic antimicrobial resistance (AMR) prediction in *Klebsiella pneumoniae*.

---

## Overview

Predicting antimicrobial resistance directly from genomic sequences is frequently compromised by **clonal data leakage**: isolates belonging to the same Sequence Type (ST) or clonal lineage share near-identical resistomes and genetic backgrounds. When randomly split between training and evaluation sets, standard models exploit lineage markers rather than learning true causal resistance determinants, leading to heavily inflated performance metrics.

To solve this:
1. **ST-Blocked Cross-Validation (Primary Evaluation)**: Isolates are grouped strictly by Sequence Type (ST) using 5-fold StratifiedGroupKFold (`random_state=4350`). Under this protocol, the attention-based Multiple Instance Learning (MIL) model achieves an **AUC of 0.8111 (95% CI: [0.7972, 0.8250])**, and the hybrid MIL + XGBoost model obtains an **AUC of 0.7981 (95% CI: [0.7837, 0.8125])**.
2. **Random-Split Leakage Benchmark**: Under conventional random partitioning (3,848 / 770 test set), the model overestimates performance with an **AUC of 0.9621 (95% CI: [0.9500, 0.9743])**, demonstrating a statistically significant performance drop ($\Delta\text{AUC} = -0.1562$, $p = 0.0014$) when clonal structure is properly controlled.

---

## Repository Structure

```
├── data/                                # Isolate metadata, manifests, and fold splits
│   ├── dataset_manifest_colab.csv       # Master cohort (3,708 isolates) with STs and splits
│   ├── Tabla_Maestra_ST_5Fold.xlsx      # Master 5-fold ST-blocked cross-validation table
│   ├── cepasv2.xlsx                     # Complete BV-BRC raw metadata table (4,869 isolates)
│   ├── Auditoria_Final_Armonizada.xlsx  # Resistance mechanisms audit table (854 isolates)
│   └── README.md
├── notebooks/                           # Step-by-step Jupyter notebooks (Colab & local ready)
│   ├── 01_prepare_dataset.ipynb         # Raw RGI filtering and ST-blocked split curation
│   ├── 02_ST_blocked_validation.ipynb   # 5-fold ST-blocked cross-validation training
│   ├── 03_MIL_XGBoost_OOF_Generation.ipynb # Out-of-fold (OOF) prediction generation
│   ├── 04_random_split_benchmark.ipynb  # Random-split comparator & false-negative profiling
│   ├── 05_analysis_and_figures.ipynb    # Model interpretability, SHAP, and PR curves
│   └── README.md
├── results/                             # Model predictions and publication-ready figures
│   ├── oof_predictions_pooled.csv       # 3,702 pooled out-of-fold cross-validation predictions
│   ├── random_split_predictions.csv     # 770 random split test predictions
│   ├── figures/                         # Generated high-resolution publication figures
│   │   ├── Figure_AUC_Comparison.png    # Primary clonal leakage comparison figure (PNG & PDF)
│   │   ├── Figure_AUC_OptionA_Subset.png# ROC curve vs. biological rule on 487 subset
│   │   └── Figure_AUC_OptionB_Full.png  # Full OOF ROC curves and relative performance drop
│   └── README.md
├── scripts/                             # Standalone scripts to reproduce all paper results
│   ├── compute_stats.py                 # Reproduces Table 1, DeLong CIs, VME/ME error rates
│   ├── figure_auc_comparison.py         # Generates Figure_AUC_Comparison (PNG/PDF)
│   ├── plot_paper_figure_options.py     # Generates ROC curve comparisons and bar charts
│   ├── run_mcnemar.py                   # Executes McNemar test (ML vs. biological rule)
│   ├── test_bootstrap.py                # Performs ST-level cluster bootstrap hypothesis tests
│   └── README.md
├── src/                                 # Modular Python package
│   ├── model.py                         # NTv3MultiHeadMIL and FocalLoss implementations
│   ├── metrics.py                       # DeLong variance, Clopper-Pearson CI, cluster bootstrap
│   └── __init__.py
├── supplementary/                       # Supplementary materials and lab checklists
│   ├── Supplementary_Figure_S4_Cohort_Flow.svg # Cohort flow diagram
│   ├── Final_Denominators.csv           # Reconciled cohort sizes across all analysis stages
│   ├── Reconciliation_3708_vs_3702.csv  # Isolate-level master-to-OOF traceability
│   ├── de_la_Fuente_Lab_Checklist.pdf   # Completed computational reproducibility checklist
│   └── README.md
├── requirements.txt                     # Pinned Python package dependencies
├── environment.yml                      # Conda / Mamba environment specification
├── LICENSE                              # MIT Open Source License
└── README.md
```

---

## Installation & Environment Setup

Python **3.12** is recommended.

### Option A: Using Conda / Mamba (Recommended)
```bash
conda env create -f environment.yml
conda activate klebsiella-mil
```

### Option B: Using Pip
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quick Reproduction (1-Command Verification)

All figures, statistical tests, and tables reported in the manuscript can be verified immediately using the standalone reproduction scripts:

```bash
# 1. Compute AUCs, DeLong confidence intervals, and VME/ME error rates (Table 1)
python scripts/compute_stats.py

# 2. Generate Figure_AUC_Comparison (.png and .pdf) in results/figures/
python scripts/figure_auc_comparison.py

# 3. Generate ROC Curves and Baseline comparisons (Option A & B)
python scripts/plot_paper_figure_options.py

# 4. Run McNemar test against the carbapenemase biological rule baseline
python scripts/run_mcnemar.py

# 5. Run cluster bootstrap across the 405 Sequence Type clusters
python scripts/test_bootstrap.py
```

### Key Verified Results

| Protocol / Model | Cohort $N$ | ROC-AUC | 95% Confidence Interval | VME Rate | ME Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ST-Blocked CV (MIL-only)** | 3,702 | **0.8111** | [0.7972, 0.8250] | 34.1% | 20.4% |
| **ST-Blocked CV (MIL + XGBoost)** | 3,702 | **0.7981** | [0.7837, 0.8125] | 53.6% | 12.3% |
| **Random Split Benchmark (Leakage)** | 770 | **0.9621** | [0.9500, 0.9743] | 7.9% | 4.8% |
| **Biological Rule Baseline** | 487 | — | — | 0.0% | 36.1% |

- **Paired DeLong Test (MIL+XGBoost vs MIL-only)**: $\Delta\text{AUC} = -0.0130$, $z = -2.7535$, $p = 0.0059$.
- **Clonal Leakage Effect**: $\Delta\text{AUC} = -0.1562$ (16.2% drop, $p = 0.0014$, one-sided $t$-test).

---

## Data Availability & Model Weights

- **Processed Data**: All metadata, manifests, ST labels, and out-of-fold predictions required to reproduce the paper's analytical findings are included directly in `data/` and `results/`.
- **Raw Sequence Data**: Raw genomic assemblies can be downloaded directly from [BV-BRC](https://www.bv-brc.org/) using the accessions listed in `data/cepasv2.xlsx` and `data/dataset_manifest_colab.csv`. RGI (v6.0.3) was executed against the CARD database (v4.0.122).
- **Pretrained Foundation Model**: The nucleotide embedding backbone uses `InstaDeepAI/NTV3_100M_post` available via Hugging Face.
- **Trained Model Weights**: Trained PyTorch checkpoints (`modelo_MIL_v5.1_best_auc.pth`, ~1.3 GB) exceed standard GitHub repository limits (< 100 MB). Checkpoints are deposited on Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22753637.svg)](https://doi.org/10.5281/zenodo.22753637) and can be placed in `models/` for full inference reproduction.

---

## Authors

- **Adolfo Morales Tena** <sup>1</sup> — Independent Researcher
- **Hanqun Cao** <sup>2</sup> — Machine Biology Group, University of Pennsylvania
- **Cesar de la Fuente-Nunez** <sup>2</sup> — Machine Biology Group, University of Pennsylvania *(corresponding author)*
- **Carlos Oscar Sorzano** <sup>3</sup> — Biocomputing Unit, Centro Nacional de Biotecnología (CNB-CSIC), Madrid, Spain

<sup>1</sup> Independent Researcher  
<sup>2</sup> Machine Biology Group, Departments of Psychiatry and Microbiology, Institute for Biomedical Informatics, Institute for Translational Medicine and Therapeutics, Perelman School of Medicine, University of Pennsylvania, Philadelphia, PA 19104, USA; Departments of Bioengineering and Chemical and Biomolecular Engineering, School of Engineering and Applied Science, University of Pennsylvania, Philadelphia, PA, USA; Department of Chemistry, School of Arts and Sciences, University of Pennsylvania, Philadelphia, PA, USA; Penn Institute for Computational Science, University of Pennsylvania, Philadelphia, PA, USA.  
<sup>3</sup> Biocomputing Unit, Centro Nacional de Biotecnología (CNB-CSIC), Madrid, Spain

If you use this code, please cite the associated paper (see [CITATION.cff](CITATION.cff)).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
