# Core Source Code (`src/`)

This directory contains modularized, reusable Python components used across notebooks and standalone reproduction scripts:

- `model.py`:
  - `NTv3MultiHeadMIL`: Multiple Instance Learning (MIL) model with multi-head attention pooling built over Nucleotide Transformer embeddings.
  - `FocalLoss`: Focal loss implementation for class imbalance in binary resistance classification.
- `metrics.py`:
  - `delong_roc_variance()`: DeLong method for computing ROC-AUC variance.
  - `delong_roc_covariance()`: DeLong method for computing covariance between paired ROC-AUC curves.
  - `compare_two_aucs()`: Paired DeLong statistical test for model comparison.
  - `clopper_pearson_ci()`: Exact binomial confidence intervals (sensitivity, specificity, VME, ME).
  - `cluster_bootstrap_compare()`: Cluster-bootstrap resampling by Sequence Type (ST).
