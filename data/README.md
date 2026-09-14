# Data Directory (`data/`)

This directory contains metadata manifests, accession identifiers, sequence type (ST) assignments, and validation fold splits.

## Files

1. **`dataset_manifest_colab.csv`** (3,708 rows):
   - Curated master cohort of *Klebsiella pneumoniae* isolates.
   - Columns:
     - `original_filename`: RGI output filename.
     - `resistance_phenotype`: Ground-truth phenotype (`Resistant` / `Susceptible`).
     - `sequence_type`: Sequence Type (ST) determined via MLST/Kleborate.
     - `dataset_partition`: Split assignment (`train` / `validation`).
     - `bvbrc_id_extraido`: BV-BRC genome identifier.
     - `gca_extraido`: NCBI GenBank assembly accession (`GCA_*`).

2. **`Tabla_Maestra_ST_5Fold.xlsx`** (3,708 rows):
   - Comprehensive master annotation table for the 5-fold ST-blocked cross-validation.
   - Columns include:
     - `validation_in_fold`: Fold assignment (1 to 5) under StratifiedGroupKFold (`random_state=4350`).
     - `audit_Mechanisms`: Detected resistance mechanisms from CARD/RGI.
     - `audit_Carbapenemase`: Binary indicator (1 = carbapenemase detected, 0 = absent).
     - Full phenotypic, isolation source, and geographic metadata.

3. **`cepasv2.xlsx`** (4,869 rows):
   - Raw metadata table extracted from BV-BRC for all isolates with explicit ertapenem testing standards, MIC/disk measurement values, collection date, country of isolation, and contig metrics.

4. **`Auditoria_Final_Armonizada.xlsx`** (854 rows):
   - Mechanism auditing subset used for false negative profiling and carbapenemase biological rule baseline benchmarking.
