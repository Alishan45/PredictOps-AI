"""
Generates the official Word document (PredictOps_AI_Week04_Project_Report.docx)
strictly adhering to the submission formatting constraints.
"""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os

doc = docx.Document()

# Adjust page margins
for section in doc.sections:
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

def set_heading(text, level, color=RGBColor(15, 23, 42)):
    p = doc.add_heading(level=level)
    run = p.add_run(text)
    run.font.color.rgb = color
    run.font.bold = True
    return p

# 1) Title, Purpose, and Short Project Overview
title_p = doc.add_paragraph()
title_run = title_p.add_run("PredictOps AI — Predictive Maintenance & Explainable ML")
title_run.font.size = Pt(22)
title_run.font.bold = True
title_run.font.color.rgb = RGBColor(15, 23, 42)
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

subtitle_p = doc.add_paragraph()
sub_run = subtitle_p.add_run("Stack & Scale | Remote Internship 2026 — AI Engineering Week 04 Final Capstone")
sub_run.font.size = Pt(13)
sub_run.font.italic = True
sub_run.font.color.rgb = RGBColor(100, 116, 139)
subtitle_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph()

set_heading("1. Project Title, Purpose & Overview", level=1)

p_purpose = doc.add_paragraph()
p_purpose.add_run("Project Purpose: ").bold = True
p_purpose.add_run(
    "In modern manufacturing plants, unplanned machine downtime causes millions of dollars in lost production and tooling damage. "
    "The purpose of PredictOps AI is to build a reliable, reproducible, and leakage-free tabular machine learning pipeline "
    "that estimates machine failure risk from continuous operational sensor telemetry (temperature, rotational speed, torque, and tool wear). "
    "Crucially, the system explains predictions through physical engineering principles without implying unwarranted certainty, "
    "empowering maintenance technicians to take timely preventative action."
)

p_overview = doc.add_paragraph()
p_overview.add_run("Project Overview: ").bold = True
p_overview.add_run(
    "PredictOps AI implements an end-to-end industrial predictive maintenance system using the public AI4I 2020 dataset (10,000 records). "
    "The architecture enforces strict data leakage defense by dropping row identifiers and causal failure modes before splitting. "
    "The pipeline synthesizes domain physical features (mechanical power in Watts, thermal dissipation differential, and tool wear strain), "
    "evaluates baseline models against candidate classifiers via 5-fold stratified cross-validation, optimizes hyperparameters for PR-AUC "
    "to counter severe class imbalance (3.39% failure rate), quantifies precision/recall operational trade-offs, and evaluates on an untouched holdout test set. "
    "The entire system is deployed via an interactive Streamlit dashboard with real-time physics attribution and sensor drift monitoring."
)

# 2) Day 1–Day 5 Task/Progress Screenshots with Short Captions
set_heading("2. Day 1–Day 5 Task & Progress Screenshots", level=1)

# Day 1
set_heading("Day 1: Data Audit, Leakage Defense & Baseline Benchmark", level=2, color=RGBColor(51, 65, 85))
p_d1 = doc.add_paragraph(
    "Audited AI4I 2020 dataset (10,000 rows, 0 missing values). Identified severe class imbalance (28:1 ratio, 3.39% failure rate). "
    "Proactively dropped 7 leakage columns (UDI, Product ID, TWF, HDF, PWF, OSF, RNF) before creating an 80/20 stratified split. "
    "Trained Monday baseline models (Dummy Majority PR-AUC: 0.0339; Logistic Regression PR-AUC: 0.4587)."
)
if os.path.exists('figures/day1_eda.png'):
    doc.add_picture('figures/day1_eda.png', width=Inches(6.2))
    cap1 = doc.add_paragraph()
    cap1_run = cap1.add_run("Figure 1: Day 1 Sensor distributions (Torque, Speed, Tool Wear) and target class imbalance (3.39% failure rate).")
    cap1_run.font.size = Pt(9.5)
    cap1_run.font.italic = True
    cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Day 2
set_heading("Day 2: Robust Training Pipeline & Candidate Classifiers", level=2, color=RGBColor(51, 65, 85))
p_d2 = doc.add_paragraph(
    "Synthesized physical domain features (Mechanical Power in Watts, Thermal Difference ΔT, and Tool Wear Strain) inside an atomic Pipeline. "
    "Evaluated two candidate classifiers (Random Forest vs HistGradientBoosting) with class weighting across 5-fold Stratified CV. "
    "Random Forest achieved 0.8879 PR-AUC and 0.9137 Balanced Accuracy, establishing a strong multi-metric lead."
)
if os.path.exists('figures/day2_pipeline_cv.png'):
    doc.add_picture('figures/day2_pipeline_cv.png', width=Inches(6.2))
    cap2 = doc.add_paragraph()
    cap2_run = cap2.add_run("Figure 2: Day 2 5-Fold Stratified Cross-Validation benchmark across class-sensitive metrics.")
    cap2_run.font.size = Pt(9.5)
    cap2_run.font.italic = True
    cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Day 3
set_heading("Day 3: Hyperparameter Tuning & Threshold Trade-off Analysis", level=2, color=RGBColor(51, 65, 85))
p_d3 = doc.add_paragraph(
    "Conducted hyperparameter grid search optimizing specifically for Average Precision (PR-AUC) under class imbalance (best CV PR-AUC: 0.8921). "
    "Analyzed Precision vs Recall curves across probability thresholds and modeled simulated plant downtime costs ($1,000 per unpredicted breakdown vs $50 per inspection). "
    "Identified threshold 0.40 as the cost-optimal operating policy."
)
if os.path.exists('figures/day3_pr_tradeoff.png'):
    doc.add_picture('figures/day3_pr_tradeoff.png', width=Inches(6.2))
    cap3 = doc.add_paragraph()
    cap3_run = cap3.add_run("Figure 3: Day 3 Precision vs Recall trade-off spectrum and simulated plant downtime cost curve.")
    cap3_run.font.size = Pt(9.5)
    cap3_run.font.italic = True
    cap3.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Day 4
set_heading("Day 4: Explainability, Spurious Correlations & Model Artifact", level=2, color=RGBColor(51, 65, 85))
p_d4 = doc.add_paragraph(
    "Computed Tree-Based MDI (Gini) and Permutation Feature Importances. Rotational Speed, Mechanical Power, Torque, and Tool Wear "
    "emerged as the primary physical precursors of impending failure. Built single-prediction physics attribution and serialized "
    "the complete pipeline to artifacts/predictops_pipeline.joblib."
)
if os.path.exists('figures/day4_feature_importance.png'):
    doc.add_picture('figures/day4_feature_importance.png', width=Inches(6.2))
    cap4 = doc.add_paragraph()
    cap4_run = cap4.add_run("Figure 4: Day 4 Global feature importance (Tree-based MDI and Permutation Importance).")
    cap4_run.font.size = Pt(9.5)
    cap4_run.font.italic = True
    cap4.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Day 5
set_heading("Day 5: Holdout Evaluation, Probability Calibration & Model Card", level=2, color=RGBColor(51, 65, 85))
p_d5 = doc.add_paragraph(
    "Evaluated the tuned pipeline once on the untouched holdout test set (2,000 machines). Achieved 98.9% Accuracy, 93.1% Balanced Accuracy, "
    "0.8766 PR-AUC, 0.9859 ROC-AUC, 86.8% Recall, and a 0.0120 Brier calibration score. Authored complete Model Card (MODEL_CARD.md)."
)
if os.path.exists('figures/day5_holdout_calibration.png'):
    doc.add_picture('figures/day5_holdout_calibration.png', width=Inches(6.2))
    cap5 = doc.add_paragraph()
    cap5_run = cap5.add_run("Figure 5: Day 5 Holdout Confusion Matrix (86.8% Recall) and Probability Calibration Reliability Curve.")
    cap5_run.font.size = Pt(9.5)
    cap5_run.font.italic = True
    cap5.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 3) Screenshots of the Final Weekly Project Output/Result
set_heading("3. Final Project Output & Streamlit Interface", level=1)
p_final = doc.add_paragraph(
    "The final deliverable is an industrial-grade Streamlit web application featuring live single-machine sensor sliders, "
    "preset scenario simulations, physics risk driver attribution, batch telemetry scoring with CSV download, and automated Kolmogorov-Smirnov drift monitoring."
)
if os.path.exists('figures/final_dashboard.png'):
    doc.add_picture('figures/final_dashboard.png', width=Inches(6.2))
    cap_f = doc.add_paragraph()
    cap_f_run = cap_f.add_run("Figure 6: Final PredictOps AI operational architecture, triage policy, and holdout performance summary.")
    cap_f_run.font.size = Pt(9.5)
    cap_f_run.font.italic = True
    cap_f.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 4) Completed Features / Deliverables
set_heading("4. Completed Features & Deliverables", level=1)

features = [
    ("Exploratory Data Audit & Leakage Defense", "Audited AI4I 2020 dataset (10,000 cycles, 3.39% failure rate); identified and safely excluded all 7 leakage attributes (UDI, Product ID, TWF, HDF, PWF, OSF, RNF) prior to splitting."),
    ("Domain Physical Feature Engineering", "Engineered mechanical power in Watts (P = tau * omega), thermal dissipation differential (Process Temp - Air Temp), and tool wear strain factor inside an atomic Pipeline."),
    ("Leakage-Safe Preprocessing Pipeline", "Constructed ColumnTransformer standardizing continuous signals and one-hot encoding product quality type without evaluation contamination."),
    ("Baseline & Candidate Model Benchmarking", "Compared Dummy Classifier and Logistic Regression baselines against Random Forest and HistGradientBoosting across 5-fold Stratified CV using class-sensitive metrics."),
    ("Hyperparameter Tuning for Class Imbalance", "Optimized Random Forest hyperparameters via GridSearchCV scoring specifically on PR-AUC (Average Precision) to counter the 28:1 imbalance."),
    ("Operational Threshold & Cost Spectrum", "Modeled precision/recall trade-offs and simulated plant downtime costs ($1,000 per breakdown vs $50 per check), identifying threshold 0.40 as the optimal preventative maintenance policy."),
    ("Global & Local Explainability", "Computed Tree-based MDI and Permutation Importance, detecting physical failure drivers (spindle power overloads, thermal drops, cutter wear) with automated single-instance explanations."),
    ("Holdout Evaluation & Probability Calibration", "Evaluated once on 2,000 untouched holdout instances, achieving 98.9% Accuracy, 93.1% Balanced Accuracy, 0.8766 PR-AUC, 86.8% Recall, and 0.0120 Brier score."),
    ("Automated Sensor Drift Monitor (Expert Challenge)", "Implemented two-sample Kolmogorov-Smirnov test monitoring continuous sensor distributions to detect sensor degradation and distribution shifts."),
    ("Interactive Streamlit Web Dashboard", "Developed 6-tab production dashboard featuring real-time sensor dials, operational scenario presets, batch CSV scoring, explanation waterfall, and live drift monitor."),
    ("Executed Reproducible Jupyter Notebook", "Authored and executed clean, top-to-bottom pipeline notebook (PredictOps_AI_Pipeline.ipynb) with all tables and figures rendered."),
    ("Production Model Card (MODEL_CARD.md)", "Documented model architecture, intended industrial use, assumptions, limitations, and operational triage recommendations.")
]

for title, desc in features:
    p_feat = doc.add_paragraph(style='List Bullet')
    p_feat.add_run(f"{title}: ").bold = True
    p_feat.add_run(desc)

# 5) Brief Explanation of Main Project Flow, Screens, and Functions
set_heading("5. Main Project Flow, Screens & Functions", level=1)

p_flow = doc.add_paragraph(
    "The PredictOps AI system operates through a structured 5-stage industrial lifecycle:\n\n"
    "1. Sensor Telemetry Ingestion & Preprocessing: Operational telemetry (air temperature, process temperature, speed, torque, tool wear) is ingested and enriched with physical indicators (power, temperature differential, wear strain), then scaled through the serialized Scikit-Learn pipeline.\n"
    "2. Machine Learning Inference: The tuned Random Forest model evaluates the multi-sensor signature and outputs a calibrated failure probability (0% to 100%).\n"
    "3. Risk Triage & Threshold Policy: Probabilities are mapped against operational thresholds (>=60% Critical, 35-60% High Risk, 15-35% Moderate, <15% Low Risk). If the score exceeds the maintenance policy threshold (default 0.40), a maintenance alert is triggered.\n"
    "4. Physics-Informed Attribution: The engine checks physical boundary conditions and highlights the dominant root cause (e.g., mechanical power spike >9,000 W or heat dissipation drop <8.6 K with worn tool).\n"
    "5. Technician Action & Fleet Management: Technicians inspect the machine and execute the prescribed preventative action (tool replacement, coolant flush, spindle bearing check), while plant managers batch-score fleets and monitor sensor drift via the Streamlit interface."
)

# 6) Weekly Learning Outcomes
set_heading("6. Weekly Learning Outcomes", level=1)

learnings = [
    ("Rigorous Leakage Prevention", "Learned how subtle post-hoc failure flags (TWF, PWF, etc.) can cause catastrophic data leakage in predictive maintenance, and mastered techniques to isolate raw operational telemetry before any modeling."),
    ("Class-Imbalance Strategy Beyond Accuracy", "Internalized why standard accuracy is deceptive in rare-event manufacturing (96.6% accuracy is trivially achievable with zero utility), and mastered optimization for PR-AUC (Average Precision), Recall, and Brier calibration."),
    ("Financial Cost Modeling of ML Decisions", "Learned to evaluate decision thresholds based on real-world industrial economics: balancing the heavy cost of unplanned catastrophic downtime ($1,000) against minor preventative inspection costs ($50)."),
    ("Physics-Informed Feature Engineering", "Discovered how incorporating first-principles domain equations (mechanical power in Watts, thermal dissipation differential) significantly elevates model predictive power (PR-AUC 0.892 vs 0.458 for raw linear baseline)."),
    ("Model Governance, Calibration & Drift Monitoring", "Gained practical experience creating standardized Model Cards, validating probability calibration curves, and implementing automated Kolmogorov-Smirnov drift tests for production reliability.")
]

for title, desc in learnings:
    p_learn = doc.add_paragraph(style='List Bullet')
    p_learn.add_run(f"{title}: ").bold = True
    p_learn.add_run(desc)

# 7) Additional Short Points (Industrial Advisory Notice)
set_heading("7. Industrial Advisory & Ethical AI Guardrails", level=1)
p_advisory = doc.add_paragraph(
    "Predictive Maintenance Advisory: Machine failure scores reflect empirical probabilities derived from historical degradation patterns. "
    "They are operational decision-support recommendations, not deterministic certainties. PredictOps AI is designed to augment maintenance teams "
    "with actionable early warnings and physical root-cause attribution, preventing unplanned downtime while ensuring human engineers retain final oversight."
)

output_filename = "PredictOps_AI_Week04_Project_Report.docx"
doc.save(output_filename)
print(f"Created {output_filename} successfully ({os.path.getsize(output_filename) / 1024:.1f} KB).")
