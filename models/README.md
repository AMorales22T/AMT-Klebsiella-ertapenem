# Model Weights

The trained PyTorch checkpoints are **not included** in this repository due to their size (~1.3 GB per fold checkpoint, ~6.5 GB total for 5 folds).

## How to Obtain the Weights

Download the trained model checkpoints from Zenodo:

> **DOI:** [10.5281/zenodo.22753637](https://doi.org/10.5281/zenodo.22753637)

After downloading, place the files in this directory following this structure:

```
models/
├── fold1_modelo_best_auc.pth
├── fold2_modelo_best_auc.pth
├── fold3_modelo_best_auc.pth
├── fold4_modelo_best_auc.pth
├── fold5_modelo_best_auc.pth
├── fold1_xgboost.pkl
├── fold2_xgboost.pkl
├── fold3_xgboost.pkl
├── fold4_xgboost.pkl
├── fold5_xgboost.pkl
└── modelo_MIL_v5.1_best_auc.pth   ← pretrained backbone used in notebook 05
```

## Model Architecture

The checkpoints correspond to the `NTv3MultiHeadMIL` architecture defined in [`src/model.py`](../src/model.py).

- **Backbone**: `InstaDeepAI/NTV3_100M_post` (Nucleotide Transformer v3, 100M parameters)
- **Attention heads**: 4 parallel attention networks (hidden dim 128)
- **Loss**: Focal Loss (α=0.55, γ=2.0)
- **Training**: 5-fold ST-blocked cross-validation (`random_state=4350`)
