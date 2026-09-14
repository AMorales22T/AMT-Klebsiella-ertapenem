# Results Directory (`results/`)

This directory contains model predictions and publication figures supporting the manuscript.

## Prediction Files

1. **`oof_predictions_pooled.csv`** (3,702 rows):
   - Out-Of-Fold (OOF) predictions across the 5 folds of ST-blocked cross-validation.
   - Columns:
     - `genome_id`: Isolate accession or identifier.
     - `y_true`: True phenotype (1 = Resistant, 0 = Susceptible).
     - `y_score`: Hybrid model (MIL + XGBoost) predicted probability.
     - `y_score_mil`: Attention-based MIL neural model predicted probability.
     - `fold`: Cross-validation fold (1 to 5).
     - `has_carbapenemase`: Binary presence of known carbapenemase gene (available for the 487 audit subset).
     - `ST`: Sequence Type cluster for ST-blocked grouping.

2. **`random_split_predictions.csv`** (770 rows):
   - Held-out test set predictions under standard random splitting (uncontrolled for clonal relatedness / demonstrating data leakage).
   - Columns:
     - `genome_id`: Isolate identifier.
     - `y_true`: True phenotype.
     - `y_score`: Predicted probability (AUC = 0.9621).

## Publication Figures (`results/figures/`)

- `Figure_AUC_Comparison.png` / `.pdf`: Primary figure showing performance drop (ΔAUC = −0.1562, p = 0.0014) caused by clonal leakage when transitioning from Random Split to ST-blocked CV.
- `Figure_AUC_OptionA_Subset.png`: Paired ROC curves and performance drop compared with the biological carbapenemase rule baseline on the 487-isolate subset.
- `Figure_AUC_OptionB_Full.png`: ROC curves for the full 3,702-isolate OOF cohort comparing Random Split, ST-Blocked MIL-only, and ST-Blocked MIL+XGBoost.
