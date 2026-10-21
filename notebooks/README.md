# Notebooks (`notebooks/`)

This directory contains the reproducible Jupyter notebooks supporting the manuscript, from raw resistome assembly to model interpretability, lineage-aware validation, and reviewer-requested architecture matching.

---

## Complete Notebook Suite

1. **`01_prepare_dataset.ipynb`**
   - **Original name:** `preparar_dataset_RGI_Kp_ertapenem_ST_Blocked.ipynb`
   - **Role:** Dataset preparation, RGI quality filtering, and creation of random & ST-blocked cross-validation splits.
   - **Dependencies:** `pandas`, `Bio` (biopython), `numpy`.

2. **`02_ST_blocked_validation.ipynb`**
   - **Original name:** `MIL_v5_1_5Fold_STblocked_Hanqun_Eval.ipynb`
   - **Role:** Primary 5-fold ST-blocked cross-validation evaluation using the attention-based Multiple Instance Learning (MIL) neural architecture (MIL-only AUC = 0.8111).
   - **Dependencies:** `torch`, `transformers`, `pandas`, `numpy`, `sklearn`.

3. **`03_MIL_XGBoost_OOF_Generation.ipynb`**
   - **Original name:** `MIL_v5_3_CV_XGBoost_LastLayer_OOF.ipynb`
   - **Role:** Generation of pooled Out-Of-Fold (OOF) cross-validation predictions (3,702 isolates) using the hybrid MIL + XGBoost model.
   - **Dependencies:** `xgboost`, `torch`, `transformers`, `pandas`, `numpy`, `sklearn`.

4. **`04_random_split_benchmark.ipynb`**
   - **Original name:** `MIL_v5_3_MultiHead_FocalLoss_Kp_ertapenem_Cosine_Penalty_xgboost__4__(1).ipynb`
   - **Role:** Clonal data leakage comparator using standard unstratified random partitioning (historical AUC = 0.9621).
   - **Dependencies:** `xgboost`, `torch`, `transformers`, `pandas`, `numpy`, `sklearn`.

5. **`05_analysis_and_figures.ipynb`**
   - **Original name:** `MIL_v5_3_Interpretability_Analysis.ipynb`
   - **Role:** Interpretability analysis (SHAP beeswarm ranking, UMAP latent clusters, PR curve, reliability calibration diagram, and error profiling).
   - **Dependencies:** `shap`, `matplotlib`, `seaborn`, `pandas`, `numpy`, `sklearn`.

6. **`06_lineage_aware_validation.ipynb`**
   - **Original name:** `Genomic_MIL_LineageAware_Validation.ipynb`
   - **Role:** Verifiable, prospectively specified lineage-aware genomic MIL validation pipeline with explicit CLI compatibility, reproducible hash verification, and paired ST-level bootstrapping.
   - **Dependencies:** `torch`, `transformers`, `xgboost`, `pandas`, `numpy`, `sklearn`.

7. **`07_architecture_matched_pipeline.ipynb`**
   - **Original name:** `AMT_Reviewer2_Comment3_ArchitectureMatched_Colab.ipynb`
   - **Role:** Architecture-matched unified evaluation addressing Reviewer 2 Comment 3 (rigorous head-to-head comparison between Random and ST-blocked splits on identical model architectures, demonstrating $\Delta\text{AUC} \approx -0.107$).
   - **Dependencies:** `torch`, `transformers`, `xgboost`, `pandas`, `numpy`, `sklearn`.

8. **`08_cohort_3848_audit.ipynb`**
   - **Original name:** `Audit_3848_Cohort_Reconstruction.ipynb`
   - **Role:** Forensic cohort reconstruction and traceability audit reconciling the initial 3,848 cohort, filtering criteria, metadata completeness, and the final 3,702 ST-blocked evaluation cohort.
   - **Dependencies:** `pandas`, `numpy`, `openpyxl`.
