from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binom

# Load data
BASE_DIR = Path(__file__).resolve().parent.parent
oof = pd.read_csv(BASE_DIR / 'results' / 'oof_predictions_pooled.csv')
audit = pd.read_excel(BASE_DIR / 'data' / 'Auditoria_Final_Armonizada.xlsx')
master = pd.read_excel(BASE_DIR / 'data' / 'Tabla_Maestra_ST_5Fold.xlsx')
cepasv2 = pd.read_excel(BASE_DIR / 'data' / 'cepasv2.xlsx')

master['clean_name'] = master['nombre_archivo_original'].str.replace('_rgi.txt', '').str.replace('.txt', '')
master_map = master.set_index('clean_name')
cepas_map = cepasv2.groupby('genome_id')['assembly_accession'].first().to_dict()
audit_map = audit.set_index('genome_id')['Carbapenemase'].to_dict()

# Reconstruct GCA accessions for OOF
gcas = []
for i, row in oof.iterrows():
    gid = str(row['genome_id'])
    gca = None
    import re
    m = re.search(r'(GCA_\d+\.\d+)', gid)
    if m:
        gca = m.group(1)
    if not gca and gid in master_map.index:
        m_row = master_map.loc[gid]
        if isinstance(m_row, pd.DataFrame):
            m_row = m_row.iloc[0]
        gca = m_row['gca_extraido']
        if pd.isna(gca) or not gca:
            bid = m_row['cepasv2_genome_id']
            gca = cepas_map.get(bid, None)
        if not gca or pd.isna(gca):
            m = re.search(r'(GCA_\d+\.\d+)', str(m_row['nombre_archivo_original']))
            gca = m.group(1) if m else None
    gcas.append(gca)

oof['gca'] = gcas

# Find rows where we have a valid GCA and that GCA exists in Auditoria
oof_audited = oof[oof['gca'].isin(audit_map.keys())].copy()
oof_audited['rule_pred'] = oof_audited['gca'].map(audit_map)

print(f"Total overlapping isolates with Auditoria: {len(oof_audited)}")

if len(oof_audited) > 0:
    # 1. Compute rule metrics on this subset
    y_true_aud = oof_audited['y_true'].values
    y_rule_aud = oof_audited['rule_pred'].values
    
    tp_r = np.sum((y_true_aud == 1) & (y_rule_aud == 1))
    tn_r = np.sum((y_true_aud == 0) & (y_rule_aud == 0))
    fp_r = np.sum((y_true_aud == 0) & (y_rule_aud == 1))
    fn_r = np.sum((y_true_aud == 1) & (y_rule_aud == 0))
    
    sens_r = tp_r / np.sum(y_true_aud == 1)
    spec_r = tn_r / np.sum(y_true_aud == 0)
    
    print("\n=== RULE BASELINE ON OVERLAP ===")
    print(f"TP: {tp_r}, TN: {tn_r}, FP: {fp_r}, FN: {fn_r}")
    print(f"Sensitivity: {sens_r:.4f}")
    print(f"Specificity: {spec_r:.4f}")
    
    # 2. Compute model metrics (threshold = 0.5) on this subset
    y_model_aud = (oof_audited['y_score'].values >= 0.5).astype(int)
    
    tp_m = np.sum((y_true_aud == 1) & (y_model_aud == 1))
    tn_m = np.sum((y_true_aud == 0) & (y_model_aud == 0))
    fp_m = np.sum((y_true_aud == 0) & (y_model_aud == 1))
    fn_m = np.sum((y_true_aud == 1) & (y_model_aud == 0))
    
    sens_m = tp_m / np.sum(y_true_aud == 1)
    spec_m = tn_m / np.sum(y_true_aud == 0)
    
    print("\n=== MODEL (MIL+XGBoost) ON OVERLAP ===")
    print(f"TP: {tp_m}, TN: {tn_m}, FP: {fp_m}, FN: {fn_m}")
    print(f"Sensitivity: {sens_m:.4f}")
    print(f"Specificity: {spec_m:.4f}")
    
    # 3. McNemar Test between Model and Rule
    # Contingency table:
    #                 Model Correct | Model Incorrect
    # Rule Correct         a               b
    # Rule Incorrect       c               d
    #
    # Wait, McNemar is typically run on the predictions themselves:
    #                 Model = 1 | Model = 0
    # Rule = 1            a           b
    # Rule = 0            c           d
    table = pd.crosstab(y_rule_aud, y_model_aud).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    print("\nContingency Table for McNemar's Test (Predictions):")
    print(table)
    
    # Extract values from 2x2 table
    b = table.loc[0, 1]
    c = table.loc[1, 0]
    p_val = binom.cdf(min(b, c), b + c, 0.5) * 2
    print(f"McNemar Test p-value: {p_val:.6f}")
    
    # Also McNemar on correctness (agreement on being correct/incorrect)
    model_correct = (y_model_aud == y_true_aud).astype(int)
    rule_correct = (y_rule_aud == y_true_aud).astype(int)
    table_corr = pd.crosstab(rule_correct, model_correct).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    print("\nContingency Table for McNemar's Test (Correctness):")
    print(table_corr)
    b_c = table_corr.loc[0, 1]
    c_c = table_corr.loc[1, 0]
    p_val_corr = binom.cdf(min(b_c, c_c), b_c + c_c, 0.5) * 2
    print(f"McNemar Test on Correctness p-value: {p_val_corr:.6f}")
