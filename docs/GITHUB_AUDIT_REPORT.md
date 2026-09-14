# GITHUB AUDIT REPORT

## 1. RESUMEN
Se realizó una auditoría de la carpeta `/mnt/windows/Users/AMT22/Downloads/` y `/mnt/windows/Users/AMT22/Downloads/cell paper/` para identificar el código correspondiente al artículo sobre atención-based MIL y validación ST-blocked en *Klebsiella pneumoniae*. 
Se inventariaron 257 archivos `.ipynb`. Mediante el mapeo de métricas clave (3708, 3848, 0.8111, 0.7981, 0.9621, 3702, etc.), se lograron aislar los 5 notebooks definitivos y los archivos de predicciones y metadatos necesarios para publicarlos en GitHub.

## 2. ARCHIVOS SELECCIONADOS
Se creó una estructura de repositorio en `Github_subir_codigo` que incluye `notebooks/`, `scripts/`, `data/`, `results/`, `src/` y `supplementary/`.

## 3. NOTEBOOKS SELECCIONADOS
1. `01_prepare_dataset.ipynb` (Original: `assamble/preparar_dataset_RGI_Kp_ertapenem_ST_Blocked.ipynb`)
2. `02_ST_blocked_validation.ipynb` (Original: `assamble/MIL_v5_1_5Fold_STblocked_Hanqun_Eval.ipynb`)
3. `03_MIL_XGBoost_OOF_Generation.ipynb` (Original: `MIL_v5_3_CV_XGBoost_LastLayer_OOF.ipynb`)
4. `04_random_split_benchmark.ipynb` (Original: `MIL_v5_3_MultiHead_FocalLoss_Kp_ertapenem_Cosine_Penalty_xgboost__4__(1).ipynb`)
5. `05_analysis_and_figures.ipynb` (Original: `assamble/MIL_v5_3_Interpretability_Analysis.ipynb`)

## 4. NOTEBOOKS DESCARTADOS Y POR QUÉ
- **Históricos y backups:** `MIL_v5_1_5Fold_STblocked_Hanqun_Eval_v2`, `...v3`, `MIL_v5_3_CV_XGBoost_LastLayer_OOF_backup.ipynb`, etc. Generaban solapamiento y representaban estados intermedios.
- **Auditorías y scripts fallidos:** `analisis_what_is_not_promising.ipynb`, `Auditoria_al_modelo_de_flaming_AI.ipynb` no corresponden a la vía principal del manuscrito.
- **Otros proyectos:** `bindingdb_v2.ipynb`, `protein_ligand...`, `LivePortrait...` no tienen relación con *Klebsiella pneumoniae*.
- **Drafts abandonados:** `entrenamiento_MIL_v4_NTv3...` (Versiones anteriores a la v5).

## 5. SCRIPTS SELECCIONADOS
Se incluyen solo aquellos scripts directamente relacionados con la generación de figuras y validación de las métricas OOF:
- `figure_auc_comparison.py`
- `plot_paper_figure_options.py`
- `compute_stats.py`
- `run_mcnemar.py`
- `test_bootstrap.py`

*(Se descartaron los de manipulación de Word, XML o revisiones editoriales como `update_manuscript.py`)*.

## 6. DATOS SELECCIONADOS
- `dataset_manifest_colab.csv` y `cepasv2.xlsx`: Solo contienen identificadores (genome_id, country, phenotype) e información estadística/bioinformática pública.
- `oof_predictions_pooled.csv` y `random_split_predictions.csv`: Contienen los resultados netos que soportan el 0.8111 y 0.9621.

## 7. DATOS QUE NO DEBEN SUBIRSE
- **Metadatos sensibles o masivos:** Archivos descargados de BV-BRC o datos contig crudos no se subieron para evitar violar políticas de GitHub (límites de tamaño) y redistribución.
- **Auditoría interna exhaustiva:** `Auditoria_Final_Armonizada.xlsx`, `Mechanism_Error_Analysis.xlsx`. (Mejor dejarlos como recursos suplementarios en Zenodo si fuera necesario, pero no en el repo directo al tener gran tamaño y posible ruido interno).

## 8. DEPENDENCIAS
Identificadas a través del análisis de celdas y scripts:
`transformers`, `captum`, `biopython`, `numpy`, `shap`, `joblib`, `xgboost`, `scikit-learn`, `matplotlib`, `seaborn`, `umap-learn`, `torch`, `pandas`, `scipy`.

## 9. VERSIONES CONFIRMADAS
- `numpy==2.0.2` (Aparece de forma explícita en una instalación `pip` dentro de un notebook).

## 10. VERSIONES NO CONFIRMADAS
- Resto de librerías (`torch`, `xgboost`, `transformers`, etc.). Se documentan sin versión explícita en `requirements.txt`.

## 11. RUTAS ABSOLUTAS QUE HAY QUE LIMPIAR
Múltiples notebooks contienen referencias a Google Drive (Colab) que deben cambiarse a relativas. Ejemplos:
- `/content/drive/MyDrive/rgi_Klebsiella_pneumoniae_ertapenem`
- `/content/Auditoria_Final_Armonizada.xlsx`
- `/content/cepasv2.xlsx`

## 12. SECRETOS/INFORMACIÓN PRIVADA ENCONTRADA
No se extrajeron API keys ni passwords directos en los códigos base exportados. Solo existen mensajes de output de `HuggingFace` solicitando loguearse ("The secret HF_TOKEN does not exist in your Colab secrets"), indicando el uso de modelos en el Hub, pero la key en sí no está embebida.

## 13. RESULTADOS QUE CADA NOTEBOOK REPRODUCE
- **02_ST_blocked_validation.ipynb**: Reproduce el entrenamiento ST-blocked. (Asociado al AUC 0.8111 y 494 falsos negativos).
- **03_MIL_XGBoost_OOF_Generation.ipynb**: Reproduce las `3,702` predicciones OOF agrupadas ST-blocked.
- **04_random_split_benchmark.ipynb**: Reproduce el modelo benchmark con leakage, alcanzando `0.9621` de AUC.
- **05_analysis_and_figures.ipynb**: Reproduce las visualizaciones y stats clave extraídas del dataframe de 1448 genomas resistentes.

## 14. ELEMENTOS QUE TODAVÍA REQUIEREN VERIFICACIÓN
- Evaluar si `cepasv2.xlsx` contiene alguna columna que la revista prohíba publicar en repositorios de código y deba estar puramente en Zenodo.
- Cambiar rutas en `.ipynb` de `/content/...` a `./data/...`.

## 15. ESTRUCTURA FINAL PROPUESTA DEL REPOSITORIO
```
Github_subir_codigo/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── data/
│   ├── cepasv2.xlsx
│   └── dataset_manifest_colab.csv
├── notebooks/
│   ├── 01_prepare_dataset.ipynb
│   ├── 02_ST_blocked_validation.ipynb
│   ├── 03_MIL_XGBoost_OOF_Generation.ipynb
│   ├── 04_random_split_benchmark.ipynb
│   ├── 05_analysis_and_figures.ipynb
│   └── README.md
├── results/
│   ├── oof_predictions_pooled.csv
│   └── random_split_predictions.csv
├── scripts/
│   ├── compute_stats.py
│   ├── figure_auc_comparison.py
│   ├── plot_paper_figure_options.py
│   ├── run_mcnemar.py
│   └── test_bootstrap.py
├── src/
│   └── README.md
└── supplementary/
    └── README.md
```

## TABLA RESUMEN

| Archivo | ¿Subir? | Razón | Evidencia | Acción |
|---|---|---|---|---|
| `preparar_dataset_RGI_Kp_ertapenem_ST_Blocked.ipynb` | SÍ | Pipeline inicial de partición | Divide los 3848 (3078/770) | Copiado como `01_prepare_dataset.ipynb` |
| `MIL_v5_1_5Fold_STblocked_Hanqun_Eval.ipynb` | SÍ | Evaluación primaria ST-blocked | Imprime los 494 errores | Copiado como `02_ST_blocked_validation.ipynb` |
| `MIL_v5_3_CV_XGBoost_LastLayer_OOF.ipynb` | SÍ | Genera el OOF pooled | Referencia `oof_predictions_pooled` | Copiado como `03_MIL_XGBoost_OOF_Generation.ipynb` |
| `MIL_v5_3_MultiHead...xgboost__4__(1).ipynb` | SÍ | Benchmark Random Split | Contiene resultado `0.9621` | Copiado como `04_random_split_benchmark.ipynb` |
| `MIL_v5_3_Interpretability_Analysis.ipynb` | SÍ | Análisis de las discrepancias | Aparecen 1448, 776, 3702, etc. | Copiado como `05_analysis_and_figures.ipynb` |
| `dataset_manifest_colab.csv` | SÍ | IDs necesarios | Headers solo tienen data bioinformática | Copiado a `data/` |
| `cepasv2.xlsx` | SÍ* | Metadatos útiles | *Requiere verificación de no tener personal info, parece seguro | Copiado a `data/` |
| `oof_predictions_pooled.csv` | SÍ | Datos tabulares de resultados | 3703 líneas coinciden con el paper | Copiado a `results/` |
| `Auditoria_Final_Armonizada.xlsx` | NO | Archivo de auditoría gigante | Contiene 854/1383 flags combinados | Descartado de GitHub |
| `bindingdb_v2.ipynb` | NO | Proyecto no relacionado | Relativo a `bindingdb` | Descartado |
| `update_manuscript.py` | NO | Script editor | Usa `docx` manipulation | Descartado |
