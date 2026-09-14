# Notebooks

1. **01_prepare_dataset.ipynb**
   - **Nombre original:** `preparar_dataset_RGI_Kp_ertapenem_ST_Blocked.ipynb`
   - **Función:** Preparación de los datos y creación de splits (random y ST-blocked).
   - **Dependencias:** `pandas`, `Bio` (biopython), `numpy`.
   - **Por qué se seleccionó:** Es el notebook base que toma los archivos fuente de los aislamientos y organiza el dataset (incluyendo referencias a splits como 3078 / 770).

2. **02_ST_blocked_validation.ipynb**
   - **Nombre original:** `MIL_v5_1_5Fold_STblocked_Hanqun_Eval.ipynb`
   - **Función:** Evaluación primaria mediante validación cruzada (5-Fold ST-blocked) usando un modelo de atención MIL.
   - **Dependencias:** `torch`, `transformers`, `pandas`, `numpy`, `sklearn`.
   - **Por qué se seleccionó:** Ejecuta la evaluación ST-blocked, que es la evaluación primaria descrita en el paper, y reporta el AUC de MIL-only (0.8111).

3. **03_MIL_XGBoost_OOF_Generation.ipynb**
   - **Nombre original:** `MIL_v5_3_CV_XGBoost_LastLayer_OOF.ipynb`
   - **Función:** Generación de predicciones Out-Of-Fold (OOF) para el conjunto completo (3,702 predicciones).
   - **Dependencias:** `xgboost`, `pandas`, `numpy`, `sklearn`, `torch`, `transformers`.
   - **Por qué se seleccionó:** Genera el archivo `oof_predictions_pooled.csv` que sustenta los resultados OOF del paper.

4. **04_random_split_benchmark.ipynb**
   - **Nombre original:** `MIL_v5_3_MultiHead_FocalLoss_Kp_ertapenem_Cosine_Penalty_xgboost__4__(1).ipynb`
   - **Función:** Comparador de data leakage usando un benchmark de random split (legado).
   - **Dependencias:** `xgboost`, `pandas`, `numpy`, `sklearn`, `torch`, `transformers`.
   - **Por qué se seleccionó:** Contiene la evaluación MIL+XGBoost random-split que alcanza el AUC de 0.9621, usado como comparador en el estudio.

5. **05_analysis_and_figures.ipynb**
   - **Nombre original:** `MIL_v5_3_Interpretability_Analysis.ipynb`
   - **Función:** Análisis de interpretabilidad y validación de métricas.
   - **Dependencias:** `shap`, `matplotlib`, `seaborn`, `pandas`, `numpy`.
   - **Por qué se seleccionó:** Notebook de análisis central que engloba la mayor parte de las métricas (494 FN, 776 FN, 1448 resistentes, etc.) mencionadas en el paper.

**Notas sobre versiones descartadas:**
Se excluyeron versiones históricas (ej. `v2`, `v3`, `v4`), versiones `_backup` y notebooks destinados a otros papers (`bindingdb_v2.ipynb`, `Flaming_AI.ipynb`).
