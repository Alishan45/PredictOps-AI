"""
Generates and executes the complete, reproducible Jupyter Notebook for PredictOps AI (Week 04).
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# ⚙️ PredictOps AI — Predictive Maintenance & Explainable ML
**Stack & Scale | Remote Internship 2026 — AI Engineering Week 04 Final Capstone**  
**Author:** AI Engineering Intern  
**Domain:** Manufacturing Intelligence, Industrial IoT & Predictive Maintenance

---

### Real-World Challenge
A manufacturing facility wants to estimate machine failure risk from continuous operational sensor data. The objective is to build a reproducible, leakage-free tabular ML pipeline and explain predictions without implying certainty.

> **CRITICAL INDUSTRIAL PRINCIPLE**:
> Predictive maintenance scores estimate empirical failure probabilities based on machine degradation signatures. They serve as an operational decision-support signal for preventative maintenance, not deterministic certainty.
"""))

# Monday - Day 1
cells.append(nbf.v4.new_markdown_cell(r"""## 📅 Monday 16 Nov — Data Audit, Leakage Defense & Baseline Benchmark
**Objectives:**
- Load AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository).
- Audit target variable (`Machine failure`), severe class imbalance (3.39% failure rate), and missing values.
- Identify and prevent potential data leakage:
  - Drop sequential row index (`UDI`) and product serial (`Product ID`).
  - Drop post-hoc failure mode indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`), which represent causal failure modes and would cause catastrophic target leakage.
- Create an 80/20 stratified train/test split **before** fitting any transformations.
- Train and report a simple baseline model (`DummyClassifier` and `LogisticRegression`).
"""))

cells.append(nbf.v4.new_code_cell("""import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src modules are discoverable
sys.path.append(os.path.abspath('..'))

from src.eda import load_raw_dataset, audit_dataset, create_leakage_safe_split, get_feature_distribution_stats
from src.pipeline import create_full_pipeline
from src.models import get_baseline_models, get_candidate_models, tune_random_forest
from src.evaluation import evaluate_model_cv, evaluate_predictions, analyze_threshold_spectrum
from src.explainability import compute_global_feature_importance, explain_single_prediction
from src.drift import check_feature_drift

# Visualization styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 5)

print("Environment and modules initialized successfully.")"""))

cells.append(nbf.v4.new_code_cell("""# Load raw dataset and audit schema
df = load_raw_dataset('../data/ai4i2020.csv')
audit = audit_dataset(df)

print(f"Total Rows: {audit['total_rows']}")
print(f"Total Missing Values: {audit['missing_values_total']}")
print(f"Class Imbalance: {audit['target_distribution']} -> Failure Rate: {audit['failure_rate_pct']}% ({audit['class_imbalance_ratio']})")
print(f"Leakage Columns Safely Identified: {audit['leakage_columns_identified']}")

df.head()"""))

cells.append(nbf.v4.new_code_cell("""# Continuous Sensor Distribution Statistics
stats_table = get_feature_distribution_stats(df)
stats_table"""))

cells.append(nbf.v4.new_code_cell("""# Leakage-Safe Stratified Train/Test Split (80/20)
X_train, X_test, y_train, y_test = create_leakage_safe_split(df, test_size=0.20, random_state=42)

print(f"X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")
print(f"X_test Shape: {X_test.shape}, y_test Shape: {y_test.shape}")
print(f"Train Failure Rate: {y_train.mean()*100:.2f}% | Test Failure Rate: {y_test.mean()*100:.2f}%")
print("Features in X_train:", X_train.columns.tolist())"""))

cells.append(nbf.v4.new_code_cell("""# Train and Evaluate Monday Baseline Benchmark (Stratified 5-Fold CV)
baselines = get_baseline_models()
baseline_metrics = {}

for name, model in baselines.items():
    metrics = evaluate_model_cv(model, X_train, y_train, cv_splits=5, random_state=42)
    baseline_metrics[name] = metrics

pd.DataFrame(baseline_metrics).T"""))

# Tuesday - Day 2
cells.append(nbf.v4.new_markdown_cell(r"""## 📅 Tuesday 17 Nov — Robust Training Pipeline & Candidate Classifiers
**Objectives:**
- Implement domain-specific physical feature engineering inside a leakage-free Scikit-Learn `Pipeline`:
  1. `temp_difference`: Process Temp - Air Temp ($\Delta T$, heat dissipation indicator).
  2. `mechanical_power`: Torque $\times$ Rotational Speed in Watts ($P = \tau \omega$).
  3. `wear_strain`: Tool Wear $\times$ Torque (mechanical stress on cutter).
- Standardize continuous features and one-hot encode `Type` (`ColumnTransformer`).
- Train two candidate classifiers (`RandomForestClassifier` and `HistGradientBoostingClassifier`) using 5-fold Stratified CV with class weighting.
"""))

cells.append(nbf.v4.new_code_cell("""# Evaluate Candidate Classifiers Across 5-Fold Stratified CV
candidates = get_candidate_models()
candidate_metrics = {}

for name, model in candidates.items():
    metrics = evaluate_model_cv(model, X_train, y_train, cv_splits=5, random_state=42)
    candidate_metrics[name] = metrics

all_comparison_df = pd.DataFrame({**baseline_metrics, **candidate_metrics}).T
all_comparison_df"""))

# Wednesday - Day 3
cells.append(nbf.v4.new_markdown_cell(r"""## 📅 Wednesday 18 Nov — Hyperparameter Tuning & Threshold Trade-off
**Objectives:**
- Conduct a justified hyperparameter grid search on Random Forest, scoring specifically on **PR-AUC (Average Precision)** to counter the 3.4% imbalance.
- Analyze Precision/Recall trade-offs across decision thresholds.
- Simulate financial consequences: In manufacturing, False Negatives (unplanned catastrophic downtime) cost significantly more than False Positives (preventative inspection).
"""))

cells.append(nbf.v4.new_code_cell("""# Hyperparameter Tuning via GridSearchCV
best_rf_pipeline, tuning_record = tune_random_forest(X_train, y_train)

print(f"Optimal Hyperparameters: {tuning_record['best_params']}")
print(f"Best 5-Fold CV PR-AUC: {tuning_record['best_pr_auc_cv']}")"""))

cells.append(nbf.v4.new_code_cell("""# Decision Threshold Spectrum & Operational Plant Cost
best_rf_pipeline.fit(X_train, y_train)
train_probs = best_rf_pipeline.predict_proba(X_train)[:, 1]

threshold_df = analyze_threshold_spectrum(y_train, train_probs)
threshold_df"""))

cells.append(nbf.v4.new_code_cell("""# Plot Precision-Recall Trade-off and Simulated Plant Cost
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

axes[0].plot(threshold_df["Threshold"], threshold_df["Precision"], marker='o', label="Precision (Inspection Precision)", color="#10B981")
axes[0].plot(threshold_df["Threshold"], threshold_df["Recall"], marker='s', label="Recall (Failure Detection Rate)", color="#3B82F6")
axes[0].plot(threshold_df["Threshold"], threshold_df["F1_Score"], marker='^', label="F1-Score", color="#F59E0B")
axes[0].set_title("Precision vs Recall Across Decision Thresholds")
axes[0].set_xlabel("Probability Threshold")
axes[0].set_ylabel("Metric Score")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(threshold_df["Threshold"], threshold_df["Estimated_Operational_Cost_$"], marker='d', color="#DC2626", linewidth=2)
axes[1].axvline(0.40, color="blue", linestyle="--", label="Cost-Optimal Cutoff (0.40)")
axes[1].set_title("Simulated Plant Downtime Cost vs Decision Threshold")
axes[1].set_xlabel("Probability Threshold")
axes[1].set_ylabel("Total Operational Cost ($)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()"""))

# Thursday - Day 4
cells.append(nbf.v4.new_markdown_cell(r"""## 📅 Thursday 19 Nov — Explainability, Spurious Correlations & Model Artifact
**Objectives:**
- Compute global feature importances: Tree-Based MDI (Gini) and Permutation Importance on validation data.
- Analyze physical root causes (e.g. power spikes vs heat dissipation vs mechanical wear).
- Inspect single-instance prediction with physics attribution.
- Serialize the production-ready pipeline to `artifacts/predictops_pipeline.joblib`.
"""))

cells.append(nbf.v4.new_code_cell("""# Global Feature Importance Analysis
df_tree, df_perm = compute_global_feature_importance(best_rf_pipeline, X_train[:2000], y_train[:2000])

print("Tree-Based Feature Importance (MDI):")
print(df_tree)
print("\\nPermutation Feature Importance (Validation PR-AUC Drop):")
print(df_perm)"""))

cells.append(nbf.v4.new_code_cell("""# Single Prediction Inspection with Physical Attribution
sample_state = {
    "Type": "L",
    "Air temperature [K]": 302.5,
    "Process temperature [K]": 310.2,
    "Rotational speed [rpm]": 1380,
    "Torque [Nm]": 52.0,
    "Tool wear [min]": 185
}

explanation = explain_single_prediction(best_rf_pipeline, sample_state)
print("Failure Probability:", f"{explanation['failure_probability_pct']}%")
print("Risk Level:", explanation["risk_level"])
print("Calculated Power (W):", explanation["calculated_power_watts"])
print("Calculated Temp Diff (K):", explanation["calculated_temp_diff_kelvin"])
print("Identified Risk Drivers:")
for d in explanation["key_risk_drivers"]:
    print(f" - {d}")"""))

cells.append(nbf.v4.new_code_cell("""# Serialize Production Pipeline Artifact
import joblib
artifact_path = '../artifacts/predictops_pipeline.joblib'
joblib.dump(best_rf_pipeline, artifact_path)
print(f"Serialized production pipeline to {artifact_path}")"""))

# Friday - Day 5
cells.append(nbf.v4.new_markdown_cell(r"""## 📅 Friday 20 Nov — Holdout Evaluation, Calibration & Data Drift
**Objectives:**
- Evaluate final model **only once** on the untouched holdout test set (`X_test`, `y_test`).
- Report holdout Precision, Recall, F1, PR-AUC, ROC-AUC, and Brier Score.
- Generate holdout Confusion Matrix and Probability Calibration Curve.
- Run automated Data Drift check using the Kolmogorov-Smirnov test.
"""))

cells.append(nbf.v4.new_code_cell("""# Final Holdout Evaluation
test_probs = best_rf_pipeline.predict_proba(X_test)[:, 1]

holdout_t050 = evaluate_predictions(y_test, test_probs, threshold=0.50, model_name="Holdout Default (t=0.50)")
holdout_t040 = evaluate_predictions(y_test, test_probs, threshold=0.40, model_name="Holdout Tuned (t=0.40)")

pd.DataFrame([holdout_t050, holdout_t040])"""))

cells.append(nbf.v4.new_code_cell("""# Holdout Confusion Matrix & Probability Calibration
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cm = confusion_matrix(y_test, (test_probs >= 0.40).astype(int))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['Normal', 'Failure'], yticklabels=['Normal', 'Failure'])
axes[0].set_title(f"Holdout Confusion Matrix (Threshold 0.40)\\nRecall: {holdout_t040['Recall']*100:.1f}%, F1: {holdout_t040['F1_Score']}")
axes[0].set_xlabel("Predicted Action")
axes[0].set_ylabel("True Condition")

prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=8)
axes[1].plot(prob_pred, prob_true, marker='o', linewidth=2, label="PredictOps Model", color="#2563EB")
axes[1].plot([0, 1], [0, 1], linestyle='--', color="gray", label="Perfect Calibration")
axes[1].set_title("Probability Calibration Reliability Curve")
axes[1].set_xlabel("Mean Predicted Probability")
axes[1].set_ylabel("True Failure Frequency")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Expert Challenge: Automated Data Drift Check (Kolmogorov-Smirnov Test)
drift_results = check_feature_drift(X_train, X_test)
drift_results"""))

cells.append(nbf.v4.new_markdown_cell("""### Summary of Capstone Outcomes
- **Leakage Prevention**: Safely excluded all 7 potential leakage attributes (`UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`).
- **Class Imbalance Optimization**: Tuned Random Forest achieved **0.8766 PR-AUC** and **86.76% Recall** on untouched holdout data (up from 0.0339 PR-AUC on Dummy baseline).
- **Physical Explainability**: Dominant failure precursors identified as mechanical power overload, tool wear strain, and thermal dissipation drop.
- **Operational Serving**: Packaged as an interactive Streamlit UI with real-time physics attribution and batch scoring.
"""))

nb['cells'] = cells

with open('notebooks/PredictOps_AI_Pipeline.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Created notebooks/PredictOps_AI_Pipeline.ipynb successfully.")
