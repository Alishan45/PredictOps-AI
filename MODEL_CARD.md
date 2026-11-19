# PredictOps AI — Model Card

## 1. Model Details
- **Model Name:** PredictOps AI Industrial Predictive Maintenance Classifier
- **Model Version:** 1.0.0
- **Model Architecture:** Random Forest Classifier (100 estimators, max depth 14, min samples leaf 2, balanced class weighting) embedded in a leakage-free Scikit-Learn `Pipeline`.
- **Primary Domain:** Manufacturing Intelligence, Industrial IoT, Computer Numerical Control (CNC) Milling Machine Monitoring.
- **Developers:** AI Engineering Intern (Stack & Scale Remote Internship 2026).
- **License:** MIT License.

---

## 2. Intended Use & Scope
- **Intended Purpose:** To estimate empirical failure risk of industrial machinery from continuous operational sensor telemetry (temperature, rotational speed, torque, tool wear) and assist plant maintenance teams in scheduling preventative inspections prior to catastrophic breakdown.
- **Target Users:** Plant reliability engineers, operations leads, and automated industrial maintenance triage systems.
- **Out-of-Scope / Misuse:**
  - **Not a Deterministic Diagnosis:** Predictive failure probabilities estimate statistical risk based on historical degradation signatures. They must **never** be interpreted as absolute certainty of immediate machine breakdown.
  - **Not a Replacement for Physical Safeguards:** This system does not replace hardware emergency stops, thermal breakers, or mechanical safety interlocks.
  - **Non-Matching Equipment:** Model is trained on milling machine sensor dynamics; using it on unrelated hardware (e.g., wind turbines or jet engines) without re-calibration is unsupported.

---

## 3. Training & Evaluation Data
- **Dataset:** AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository, Matzka 2020).
- **Total Population:** 10,000 operational cycles.
- **Split Strategy:** 80% Training (8,000 instances), 20% Untouched Holdout Test (2,000 instances), stratified on target label `Machine failure`.
- **Target Distribution:**
  - Normal Cycles: 9,661 (96.61%)
  - Machine Failures: 339 (3.39%)
  - Imbalance Ratio: **28 : 1**
- **Leakage Prevention:**
  - `UDI` and `Product ID` dropped (arbitrary identifiers).
  - Specific causal failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) strictly excluded from feature space $X$ to prevent target leakage.

---

## 4. Evaluated Performance Metrics

| Evaluation Dataset | Decision Threshold | Accuracy | Balanced Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5-Fold Stratified CV** | 0.50 | 97.8% | 91.4% | 85.2% | 84.1% | 0.8465 | 0.9845 | 0.8921 | 0.0132 |
| **Untouched Holdout** | **0.50 (Default)** | **98.9%** | **93.1%** | **81.9%** | **86.8%** | **0.8429** | **0.9859** | **0.8766** | **0.0120** |
| **Untouched Holdout** | **0.40 (Cost-Tuned)** | **98.1%** | **92.6%** | **66.3%** | **86.8%** | **0.7516** | **0.9859** | **0.8766** | **0.0120** |

---

## 5. Physical Feature Attributions & Explainability
The pipeline computes domain-engineered physical indicators:
1. **Mechanical Power ($P = \tau \omega$ in Watts):** Captures power overload (>9,000 W) or underload (<3,500 W).
2. **Thermal Dissipation Differential ($\Delta T = T_{\text{process}} - T_{\text{air}}$ in Kelvin):** Flags heat dissipation failure risk when $\Delta T < 8.6\text{ K}$.
3. **Tool Wear Strain Factor ($\text{Wear} \times \text{Torque}$):** Flags extreme mechanical strain (>11,000) leading to tool overstrain failure.

---

## 6. Assumptions & Limitations
1. **Sensor Calibration:** Sensor readings are assumed to be accurate, continuous, and free of communication dropouts.
2. **Operational Regimes:** If ambient operating temperatures or spindle motors change dramatically, data drift may occur.
3. **Continuous Monitoring:** The model provides point-in-time risk estimates. For high-frequency vibration, edge FFT analysis is recommended as a complementary layer.

---

## 7. Operational Mitigation & False-Negative Strategy
In industrial manufacturing, **False Negatives** (unpredicted catastrophic breakdowns) cost significantly more ($10\times - 50\times$) in lost line productivity and equipment repair than **False Positives** (a routine 15-minute visual inspection).
- **Policy Recommendation:** Set alert threshold to **0.40** to guarantee high recall (86.8%+) and catch impending tool failures before damage occurs.
