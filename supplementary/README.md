# Supplementary Material (`supplementary/`)

This directory contains supplementary metadata, traceability tables, and cohort reconciliation manifests supporting the manuscript.

> **Note on Supplementary Figures**: All rendered supplementary figures (300 DPI PNG, vector TrueType PDF, and source SVG) are located in **[`results/supplementary/`](../results/supplementary/)**, generated directly via **[`scripts/supplementary/plot_supplementary_figures.py`](../scripts/supplementary/plot_supplementary_figures.py)**.

---

### Supplementary Figures Overview (located in `results/supplementary/`)
- **`Supplementary_Figure_S1_Threshold_Sweep.png` / `.pdf`**: Decision threshold sweep diagnostic (Trade-off between Recall $R$, Specificity $S$, and F1 across thresholds $[0.30, 0.70]$ on the held-out validation cohort).
- **`Supplementary_Figure_S2_Attention_Head_Specialization.png` / `.pdf`**: Multi-head attention diagnostics (dominant head distribution, attention variance, and max attention weight per head).
- **`Supplementary_Figure_S3_Error_Analysis.png` / `.pdf`**: Prediction outcome analysis (predicted resistance probability boxplot across Correct, Very Major Error [FN], and Major Error [FP] isolates, alongside validation ROC curve).
- **`Supplementary_Figure_S4_Cohort_Flow.png` / `.pdf` / `.svg`**: Cohort progression flowchart tracking isolates from raw BV-BRC retrieval (34,773) down to analyzed cohorts (3,708 master cohort; 3,702 OOF vector; 770 random split held-out set).
- **`Supplementary_Figure_S5_UMAP_Embeddings.png` / `.pdf`**: 2D UMAP projection of the 768-D genome latent space representations colored by phenotypic class.

---

### Supplementary Data & Checklists (in this directory)
- **`Final_Denominators.csv`**: Summary of cohort sizes and relationships across all evaluation stages.
- **`Reconciliation_3708_vs_3702.csv`**: Detailed isolate-level reconciliation between the 3,708 master cohort and the 3,702 out-of-fold predictions.
- **`Supplementary_Table_S2_ST_Distribution.csv` / `.xlsx`**: Distribution of the 405 Sequence Types across the study cohort.
- **`de_la_Fuente_Lab_Checklist.pdf`**: Completed Computational Data and Reproducibility Checklist (Machine Biology Group).
