# Results Directory (`results/`)

This directory contains model predictions and publication figures supporting the manuscript, cleanly partitioned into active figures (**`new/`**), supplementary figures (**`supplementary/`**), and historical figures (**`legacy/`**).

---

## Directory Organization

```
results/
├── new/                                 # Current Nature Portfolio publication figures (Figures 1–6)
│   ├── Figure1_updated.png / .pdf       # Framework & resistome architecture
│   ├── Figure2_updated.png / .pdf       # ROC curves & clonal inflation drop
│   ├── Figure3_updated.png / .pdf       # SHAP global beeswarm attribution
│   ├── Figure4_updated.png / .pdf       # blaKPC-2 Integrated Gradients CDS profile
│   ├── Figure5_updated.png / .pdf       # LOGO sensitivity & top 20 driver genes (5996 × 2511 px)
│   ├── Figure6_updated.png / .pdf       # PR curve & probability calibration analysis
│   └── Figure_ROC_Error_Analysis.png / .pdf # Outcome probability distribution & ROC comparison
├── supplementary/                       # Supplementary Figures S1 through S5 (300 DPI & TrueType PDF)
│   ├── Supplementary_Figure_S1_Threshold_Sweep.png / .pdf
│   ├── Supplementary_Figure_S2_Attention_Head_Specialization.png / .pdf
│   ├── Supplementary_Figure_S3_Error_Analysis.png / .pdf
│   ├── Supplementary_Figure_S4_Cohort_Flow.png / .pdf / .svg
│   └── Supplementary_Figure_S5_UMAP_Embeddings.png / .pdf
├── legacy/                              # Historical reference figures & baseline options
│   ├── Figure1_legacy.png / .pdf        # Pre-refinement Figure 1
│   ├── Figure2_legacy.png / .pdf        # Pre-refinement Figure 2
│   ├── Figure5_legacy.png               # Pre-refinement Figure 5
│   ├── Figure_AUC_Comparison.png / .pdf # Clonal leakage drop comparison
│   └── Figure_AUC_Paper_Ready.png / .pdf# Initial paper-ready AUC layout
├── oof_predictions_pooled.csv           # 3,702 pooled out-of-fold cross-validation predictions
└── random_split_predictions.csv         # 770 random split test predictions
```

---

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
