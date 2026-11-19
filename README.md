# ⚙️ PredictOps AI — Industrial Predictive Maintenance & Explainable ML
> **Stack & Scale | Remote Internship 2026 — AI Engineering Week 04 Final Capstone**  
> *A leakage-safe tabular machine learning pipeline, physics-informed feature engineering framework, and interactive operational triage dashboard for industrial milling machine predictive maintenance.*

---

## 📌 Executive Summary & Real-World Challenge
In modern manufacturing facilities, unexpected machine failure halts production lines, destroys costly tooling, and causes substantial financial losses. Industrial reliability teams must anticipate impending mechanical failures from continuous sensor telemetry (temperature, rotational speed, torque, and cumulative tool wear) without inundating technicians with false alarms.

### ⚠️ Core Operational Principle
> **Predictive maintenance scores estimate empirical failure probabilities based on historical degradation patterns. They are operational decision-support signals, not deterministic certainties.**  
> Predictions serve to schedule preventative maintenance before catastrophic downtime occurs, augmenting human technicians rather than replacing physical safety interlocks.

---

## 🏗️ Project Architecture & Repository Structure

```
├── app.py                                  # Production Streamlit Web Dashboard
├── data/
│   ├── ai4i2020.csv                        # AI4I 2020 Predictive Maintenance Dataset (10,000 cycles)
│   └── dataset_citation.md                 # UCI provenance, license, and schema documentation
├── artifacts/
│   ├── predictops_pipeline.joblib          # Serialized production pipeline (Scikit-Learn)
│   └── model_metadata.json                 # Comprehensive training and holdout metadata
├── figures/
│   ├── day1_eda.png                        # Day 1 EDA distributions & target imbalance
│   ├── day2_pipeline_cv.png                # Day 2 Stratified 5-fold CV benchmark chart
│   ├── day3_pr_tradeoff.png                # Day 3 Precision-Recall & plant downtime cost curve
│   ├── day4_feature_importance.png         # Day 4 Permutation & Tree MDI feature importances
│   ├── day5_holdout_calibration.png        # Day 5 Holdout Confusion Matrix & Calibration curve
│   └── final_dashboard.png                 # Day 5 Final system architecture & triage policy
├── notebooks/
│   └── PredictOps_AI_Pipeline.ipynb        # Clean, top-to-bottom executed Jupyter Notebook (246 KB)
├── src/
│   ├── __init__.py
│   ├── eda.py                              # Data audit, leakage defense & stratified splitting
│   ├── pipeline.py                         # Physics feature synthesis & ColumnTransformer pipeline
│   ├── models.py                           # Baseline & candidate classifiers, GridSearchCV tuning
│   ├── evaluation.py                       # Class-sensitive metrics, PR-AUC, cost modeling
│   ├── explainability.py                   # Tree MDI, permutation importance & single-machine physics
│   └── drift.py                            # Kolmogorov-Smirnov test for operational sensor drift
├── MODEL_CARD.md                           # Comprehensive production model card
├── PredictOps_AI_Week04_Project_Report.docx # Official Word report (with embedded visual evidence)
├── requirements.txt                        # Production dependencies
├── .gitignore
└── README.md                               # Complete documentation & case study
```

---

## 🗓️ Five-Day Capstone Implementation Roadmap

### 📅 Monday 16 Nov — Data Audit, Leakage Defense & Baseline Benchmark
- **Dataset**: AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository, Matzka 2020).
- **Target Variable**: `Machine failure` (0 = Nominal operation, 1 = Machine breakdown).
- **Severe Class Imbalance**: 9,661 Normal (96.61%) vs 339 Failure (3.39%) — **28 : 1 ratio**.
- **Rigorous Leakage Defense**:
  - `UDI` (row sequential index) and `Product ID` dropped.
  - Failure mode flags `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` describe post-hoc specific failure categories. If used as input features, they cause catastrophic data leakage. **All 7 leakage columns are strictly excluded from feature matrix $X$ before any modeling.**
- **Leakage-Safe Splitting**: 80% Train (8,000 machines) / 20% Holdout Test (2,000 machines), stratified on failure rate **prior to fitting any transformers**.
- **Baseline Models (5-Fold Stratified CV)**:
  - *Dummy Majority Classifier*: PR-AUC: 0.0339, F1: 0.0000, Recall: 0.0%.
  - *Logistic Regression (Balanced)*: PR-AUC: 0.4587, F1: 0.2716, Recall: 84.15%, Balanced Acc: 84.43%.

### 📅 Tuesday 17 Nov — Robust Training Pipeline & Candidate Classifiers
- **Domain Feature Engineering (`DomainFeatureEngineer`)**:
  1. **Mechanical Power ($P = \tau \omega$ in Watts)**: $P = \text{Torque} \times \left(\text{Speed} \times \frac{2\pi}{60}\right)$. Captures power overload (>9,000 W) and underload (<3,500 W).
  2. **Thermal Dissipation Differential ($\Delta T = T_{\text{process}} - T_{\text{air}}$ in Kelvin)**: Heat dissipation failure occurs when heat cannot escape milling cutters ($\Delta T < 8.6\text{ K}$).
  3. **Tool Wear Strain Factor**: $\text{Tool Wear [min]} \times \text{Torque [Nm]}$. Measures cumulative mechanical fatigue on cutting inserts.
- **ColumnTransformer**: Standardizes continuous features with `StandardScaler` and encodes quality `Type` via `OneHotEncoder(drop='first')`.
- **Candidate Classifiers Benchmark (5-Fold Stratified CV)**:

| Model Architecture | Balanced Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC (Primary) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Majority Class)** | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0339 |
| **Logistic Regression (Balanced)** | 0.8443 | 0.1654 | 0.8415 | 0.2716 | 0.9082 | 0.4587 |
| **HistGradientBoosting (Balanced)** | 0.9079 | **0.8950** | 0.8191 | **0.8556** | 0.9780 | 0.8885 |
| **Random Forest (Tuned Balanced)** | **0.9137** | 0.8520 | **0.8412** | 0.8465 | **0.9845** | **0.8921** |

### 📅 Wednesday 18 Nov — Hyperparameter Tuning & Threshold Trade-off
- **GridSearchCV Tuning**: Optimized specifically on **Average Precision (PR-AUC)** using 5-fold Stratified CV.
  - *Optimal Parameters*: `n_estimators=100`, `max_depth=14`, `min_samples_leaf=2`, `max_features='sqrt'`.
  - *Best CV PR-AUC*: **0.8921**.
- **Financial Downtime Cost Modeling**:
  - In a manufacturing plant, an unplanned breakdown ($FN$) costs $\approx \$1,000$ in lost line throughput, whereas an unnecessary preventative check ($FP$) costs $\approx \$50$.
  - Threshold sensitivity analysis identified **0.40** as the optimal operational policy threshold, maximizing recall while keeping false alarms manageable.

### 📅 Thursday 19 Nov — Explainability, Spurious Correlations & Model Artifact
- **Global Feature Importance**:
  - *Top MDI & Permutation Features*: Rotational Speed, Mechanical Power (W), Torque, and Tool Wear.
  - Proved that quality variant `Type` had minimal predictive power compared to physical kinematics and thermal dissipation.
- **Single-Machine Physics Attribution**: Implemented real-time boundary evaluation identifying whether risk is driven by thermal dissipation failure, mechanical overstrain, or power overload.
- **Pipeline Serialization**: Serialized complete production pipeline to `artifacts/predictops_pipeline.joblib`.

### 📅 Friday 20 Nov — Holdout Evaluation, Calibration & Model Card
- **Final Untouched Holdout Evaluation (2,000 Machines)**:
  - **Accuracy:** **98.9%**
  - **Balanced Accuracy:** **93.1%**
  - **Recall:** **86.8%** (59 out of 68 true failures detected!)
  - **Precision:** **81.9%**
  - **F1-Score:** **0.8429**
  - **ROC-AUC:** **0.9859**
  - **PR-AUC:** **0.8766**
  - **Brier Calibration Score:** **0.0120** (exceptionally well-calibrated probabilities)
- **Sensor Drift Monitor (Expert Challenge)**: Implemented two-sample Kolmogorov-Smirnov test to flag distribution shifts in live sensor streams ($p < 0.05$).
- **Model Card**: Authored comprehensive documentation in [`MODEL_CARD.md`](MODEL_CARD.md).

---

## ⚙️ Interactive Streamlit Dashboard (`app.py`)

The production application features 6 operational tabs:
1. **⚡ Live Sensor Inference**: Real-time sliders for Air Temp, Process Temp, Speed, Torque, and Tool Wear with preset operational scenarios (Power Overload, Thermal Strain, Worn Tool). Includes a dynamic gauge chart and root-cause attribution.
2. **📁 Batch Telemetry Inference**: Upload fleet CSV files, score machine health en masse, and export maintenance work queues.
3. **📊 Exploratory Sensor Audit**: Inspect schema, sensor histograms, correlation heatmap, and class imbalance.
4. **⚙️ Pipeline & Benchmarks**: Review 5-fold CV benchmark tables, Precision-Recall curves, and the plant downtime cost simulator.
5. **🔍 Explainability & Physics**: Global feature importance and physical breakdown mechanisms.
6. **🛡️ Model Card & Drift Monitor**: Full model card and live Kolmogorov-Smirnov sensor drift testing.

---

## 🚀 Running the Project Locally

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Alishan45/PredictOps-AI.git
cd PredictOps-AI

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Jupyter Notebook
```bash
cd notebooks
jupyter notebook PredictOps_AI_Pipeline.ipynb
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ How to Host on Streamlit Community Cloud (Free & 1-Click)

1. Push your repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New app"**, select your repository (`<your-username>/PredictOps-AI`), branch `main`, and main file path `app.py`.
4. Click **Deploy!**

---

## 📹 Mandatory Narrated Demonstration Script (5–8 Minutes)

When recording your video demonstration for mentors and the final capstone submission:

1. **Introduction & Challenge (0:00 - 1:00)**:
   - Introduce *PredictOps AI*: A predictive maintenance system built to estimate machine failure risk from continuous operational sensor data.
   - Emphasize the core principle: *Predictive scores are operational review signals, not deterministic certainty.*
2. **Data Audit & Leakage Defense (1:00 - 2:15)**:
   - Show Tab 3 (EDA): Explain the 28:1 class imbalance (3.39% failure rate).
   - Explain why `UDI`, `Product ID`, and causal failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) were excluded to prevent catastrophic data leakage.
3. **Physics-Informed Pipeline & Benchmarks (2:15 - 4:00)**:
   - Show Tab 4: Walk through domain feature engineering (Mechanical Power in Watts, Thermal Difference $\Delta T$, and Tool Wear Strain).
   - Compare 5-fold CV results: Show how Random Forest (PR-AUC 0.8921) dramatically outperforms the Dummy baseline (0.0339) and linear model (0.4587).
4. **Threshold Trade-offs & Economic Cost (4:00 - 5:15)**:
   - Explain the precision/recall trade-off: In manufacturing, False Negatives cost $\approx \$1,000$ in lost production, while False Positives cost $\approx \$50$ in inspection time.
   - Show why threshold 0.40 was selected as the cost-optimal operating policy.
5. **Live Inference & Explainability (5:15 - 6:45)**:
   - Demo Tab 1: Toggle the "Power Overload Spike" and "Heat Dissipation Strain" presets.
   - Show the dynamic gauge chart, calculated physical metrics, and automated risk drivers.
6. **Holdout Verification, Drift & Conclusion (6:45 - 8:00)**:
   - Show Tab 6: Walk through the final holdout metrics (98.9% Accuracy, 86.8% Recall, 0.8766 PR-AUC, 0.0120 Brier score).
   - Highlight the Kolmogorov-Smirnov drift test and conclude with key engineering learnings.

---

## 💼 LinkedIn Project Showcase Post Draft

```
🚀 Final Capstone Milestone: PredictOps AI — Industrial Predictive Maintenance & Explainable ML

Thrilled to present my Week 4 Final Capstone at Stack & Scale! In industrial manufacturing, unplanned machine downtime costs millions. Building predictive maintenance models requires far more than basic accuracy—it demands leakage defense, severe class-imbalance management, and physics-informed explainability.

🔍 What I Built:
• Data Audit & Leakage Defense: Audited 10,000 machine cycles (AI4I 2020 dataset) with a 28:1 class imbalance (3.39% failures). Proactively identified and dropped 7 causal leakage columns (UDI, Product ID, TWF, HDF, PWF, OSF, RNF) prior to splitting.
• Physics-Informed Feature Engineering: Derived first-principles physical signals inside an atomic Scikit-Learn Pipeline: Mechanical Power (P = τω in Watts), Thermal Dissipation Differential (ΔT), and Tool Wear Strain.
• Class-Sensitive Benchmarking & Tuning: Evaluated models across 5-fold Stratified CV; tuned Random Forest specifically for PR-AUC (Average Precision), achieving 0.8921 CV PR-AUC.
• Economic Cost Modeling: Quantified the trade-off between catastrophic breakdown costs ($1,000) and preventative inspection costs ($50), selecting threshold 0.40 as the optimal plant policy.
• Holdout Verification: Evaluated once on 2,000 untouched holdout machines, achieving 98.9% Accuracy, 93.1% Balanced Accuracy, 86.8% Recall, 0.8766 PR-AUC, and an exceptional 0.0120 Brier calibration score.
• Production Streamlit Dashboard & Drift Monitor: Built a full UI with live sensor dials, scenario presets, physics root-cause attribution, batch CSV scoring, and automated Kolmogorov-Smirnov sensor drift checks.

⚠️ Core Engineering Principle: Predictive maintenance scores are empirical risk signals for proactive inspection, not deterministic certainty!

#AIEngineering #MachineLearning #PredictiveMaintenance #IndustrialIoT #ExplainableAI #ScikitLearn #Streamlit #StackAndScale #FinTech
```
