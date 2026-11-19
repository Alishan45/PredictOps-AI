# Dataset Provenance & Citation

## AI4I 2020 Predictive Maintenance Dataset
- **Source**: UCI Machine Learning Repository / Matzka, S. (2020)
- **URL**: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- **Authors**: Stephan Matzka, School of Engineering - Technology and Life, HTW Berlin, Germany.
- **DOI**: 10.24432/C5HS5C
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

### Dataset Overview
- **Instances**: 10,000 real/synthetic machine operation records
- **Features**: 14 raw columns (including identifiers and failure mode indicators)
- **Target Variable**: `Machine failure` (Binary: 0 = No failure, 1 = Failure)
- **Class Balance**: 9,661 Normal (96.61%), 339 Failure (3.39%) — Severe class imbalance.

### Feature Dictionary
1. `UDI`: Unique Identifier (1 to 10,000) — *Excluded (Leakage/Identifier)*
2. `Product ID`: Product variant letter + variant serial — *Excluded (Redundant)*
3. `Type`: Product quality variant (`L` = Low 50%, `M` = Medium 30%, `H` = High 20%) — *Categorical Feature*
4. `Air temperature [K]`: Ambient sensor temperature (Kelvin, 295.3 - 304.5 K) — *Numerical Feature*
5. `Process temperature [K]`: Internal manufacturing process temperature (Kelvin, 305.7 - 313.8 K) — *Numerical Feature*
6. `Rotational speed [rpm]`: Spindle rotational speed (RPM, 1,168 - 2,886 rpm) — *Numerical Feature*
7. `Torque [Nm]`: Mechanical torque (Newton-meters, 3.8 - 76.6 Nm) — *Numerical Feature*
8. `Tool wear [min]`: Cumulative tool usage time (Minutes, 0 - 253 min) — *Numerical Feature*
9. `Machine failure`: Global failure label (0 or 1) — *Primary Supervised Target*
10. `TWF`: Tool Wear Failure mode — *Excluded from Features (Causal Leakage)*
11. `HDF`: Heat Dissipation Failure mode — *Excluded from Features (Causal Leakage)*
12. `PWF`: Power Failure mode — *Excluded from Features (Causal Leakage)*
13. `OSF`: Overstrain Failure mode — *Excluded from Features (Causal Leakage)*
14. `RNF`: Random Failure mode — *Excluded from Features (Causal Leakage)*

### Critical Leakage Note
The columns `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` are post-hoc specific failure diagnosis flags. Including them in the feature set to predict `Machine failure` constitutes direct causal data leakage. They are strictly excluded from all training and inference pipelines.
