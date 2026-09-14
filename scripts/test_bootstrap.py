from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Load OOF predictions
BASE_DIR = Path(__file__).resolve().parent.parent
oof = pd.read_csv(BASE_DIR / 'results' / 'oof_predictions_pooled.csv')

# Use ST column already present in oof_predictions_pooled.csv
if 'ST' in oof.columns and oof['ST'].notna().any():
    oof['sequence_type'] = oof['ST'].astype(str)
else:
    # Fallback to master table if present
    master_path = BASE_DIR / 'data' / 'Tabla_Maestra_ST_5Fold.xlsx'
    if master_path.exists():
        master = pd.read_excel(master_path)
        def standardize_name(name):
            if pd.isna(name): return ""
            name = str(name)
            import re
            m = re.search(r'(GCA_\d+\.\d+)', name)
            if m: return m.group(1)
            return name.replace('.txt', '').replace('_rgi', '')
        master['clean_name'] = master['nombre_archivo_original'].apply(standardize_name)
        master_map = master.set_index('clean_name')['sequence_type'].to_dict()
        master_gca_map = master.dropna(subset=['gca_extraido']).set_index('gca_extraido')['sequence_type'].to_dict()
        sts = []
        for idx, row in oof.iterrows():
            gid = str(row['genome_id'])
            clean_gid = standardize_name(gid)
            st = master_map.get(clean_gid, None)
            if st is None or pd.isna(st): st = master_gca_map.get(gid, None)
            if st is None or pd.isna(st): st = master_gca_map.get(clean_gid, None)
            sts.append(st if (st is not None and not pd.isna(st)) else "Unknown")
        oof['sequence_type'] = sts
    else:
        oof['sequence_type'] = "Unknown"

# Group by sequence_type
unique_sts = list(oof['sequence_type'].unique())
print(f"Number of unique ST clusters: {len(unique_sts)}")

# Bootstrap
np.random.seed(42)
B = 1000
diffs = []
auc_mil_list = []
auc_xgb_list = []

# Pre-group data by ST to speed up bootstrap
grouped = {st: grp for st, grp in oof.groupby('sequence_type')}

for b in range(B):
    # Sample STs with replacement
    sampled_sts = np.random.choice(unique_sts, size=len(unique_sts), replace=True)
    
    # Concatenate the dataframes for the sampled STs
    sampled_df = pd.concat([grouped[st] for st in sampled_sts], ignore_index=True)
    
    # Compute AUCs
    if len(sampled_df['y_true'].unique()) > 1:
        auc_mil = roc_auc_score(sampled_df['y_true'], sampled_df['y_score_mil'])
        auc_xgb = roc_auc_score(sampled_df['y_true'], sampled_df['y_score'])
        diffs.append(auc_mil - auc_xgb)
        auc_mil_list.append(auc_mil)
        auc_xgb_list.append(auc_xgb)

mean_diff = np.mean(diffs)
ci_lower = np.percentile(diffs, 2.5)
ci_upper = np.percentile(diffs, 97.5)
p_value = 2 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0))

print("\n=== CLUSTER BOOTSTRAP RESULTS ===")
print(f"Mean AUC (MIL-only):    {np.mean(auc_mil_list):.4f}")
print(f"Mean AUC (MIL+XGBoost): {np.mean(auc_xgb_list):.4f}")
print(f"Mean Delta AUC (MIL-only - MIL+XGBoost): {mean_diff:+.4f}")
print(f"ST-level bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"ST-level bootstrap p-value: {p_value:.6f}")
