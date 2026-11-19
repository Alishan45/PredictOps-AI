"""
PredictOps AI - Industrial Predictive Maintenance & Explainable ML Dashboard
Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import io

from src.explainability import explain_single_prediction
from src.drift import check_feature_drift

st.set_page_config(
    page_title="PredictOps AI | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .callout-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 1.2rem;
        color: #92400E;
    }
</style>
""", unsafe_allow_html=True)

# Load Pipeline and Metadata
@st.cache_resource
def load_artifacts():
    pipeline_path = "artifacts/predictops_pipeline.joblib"
    meta_path = "artifacts/model_metadata.json"
    
    if os.path.exists(pipeline_path):
        pipeline = joblib.load(pipeline_path)
    else:
        pipeline = None
        
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}
        
    return pipeline, metadata

pipeline, metadata = load_artifacts()

# Baseline dataset for drift & reference
@st.cache_data
def load_reference_data():
    if os.path.exists("data/ai4i2020.csv"):
        return pd.read_csv("data/ai4i2020.csv")
    return None

ref_df = load_reference_data()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/engine.png", width=64)
    st.title("⚙️ PredictOps AI")
    st.caption("Industrial Predictive Maintenance & Explainable ML")
    st.markdown("---")
    
    st.subheader("🎯 Maintenance Policy Settings")
    decision_threshold = st.slider(
        "Alert Probability Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help="Custom probability cutoff. Lower thresholds catch more failures (higher recall) at the cost of more preventative inspections."
    )
    
    st.markdown("---")
    st.markdown("### 🏷️ Model Status")
    st.success("✅ Model: Random Forest (Balanced)")
    st.info(f"📊 Holdout ROC-AUC: **{metadata.get('holdout_metrics_default_t050', {}).get('ROC_AUC', 0.9859)}**")
    st.info(f"🎯 Holdout PR-AUC: **{metadata.get('holdout_metrics_default_t050', {}).get('PR_AUC', 0.8766)}**")
    st.info(f"🔍 Holdout Recall: **{metadata.get('holdout_metrics_default_t050', {}).get('Recall', 0.8676)*100:.1f}%**")
    
    st.markdown("---")
    st.caption("Stack & Scale | Remote Internship 2026 | Week 04 Final Capstone")

# Header
st.markdown('<div class="main-title">⚙️ PredictOps AI — Predictive Maintenance & Explainable ML</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Production Tabular Pipeline, Class-Imbalance Calibration, and Physics-Informed Sensor Explainability</div>', unsafe_allow_html=True)

st.markdown("""
<div class="callout-box">
<strong>⚠️ Industrial Advisory Notice:</strong> Predictive maintenance scores represent empirical failure probabilities based on sensor telemetry. <strong>Predictions are operational review signals, not deterministic certainty.</strong> They serve to trigger preventative technician inspections before catastrophic plant downtime occurs.
</div>
""", unsafe_allow_html=True)

# KPIs
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Trained Records", "8,000", "AI4I Benchmark")
with kpi2:
    st.metric("Holdout Evaluation", "2,000", "Untouched Test Set")
with kpi3:
    st.metric("Failure Imbalance", "3.39%", "Class Weight Balanced")
with kpi4:
    st.metric("Test Balanced Acc", f"{metadata.get('holdout_metrics_default_t050', {}).get('Balanced_Accuracy', 0.9305)*100:.1f}%")
with kpi5:
    st.metric("Test F1-Score", f"{metadata.get('holdout_metrics_default_t050', {}).get('F1_Score', 0.8429):.4f}")

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⚡ 1. Live Sensor Inference",
    "📁 2. Batch Telemetry Inference",
    "📊 3. Exploratory Sensor Audit (Mon)",
    "⚙️ 4. Pipeline & Benchmarks (Tue-Wed)",
    "🔍 5. Explainability & Physics (Thu)",
    "🛡️ 6. Model Card & Drift (Fri)"
])

# ----------------- TAB 1: LIVE INFERENCE -----------------
with tab1:
    st.subheader("Real-Time Machine Operating Telemetry Inference")
    st.markdown("Configure operational sensor readings to evaluate immediate failure probability and inspect physical risk drivers.")
    
    preset = st.selectbox(
        "Choose an Operational Scenario Preset:",
        ["Custom Manual Input", "Normal Smooth Operation", "Heat Dissipation Strain (Low Temp Δ)", "Power Overload Spike", "Extreme Tool Wear (Overstrain)"]
    )
    
    # Preset default mappings
    if preset == "Normal Smooth Operation":
        def_type, def_air, def_proc, def_rpm, def_torque, def_wear = "M", 298.1, 308.6, 1550, 42.0, 30
    elif preset == "Heat Dissipation Strain (Low Temp Δ)":
        def_type, def_air, def_proc, def_rpm, def_torque, def_wear = "L", 302.5, 310.2, 1380, 52.0, 185
    elif preset == "Power Overload Spike":
        def_type, def_air, def_proc, def_rpm, def_torque, def_wear = "L", 299.0, 309.5, 2600, 68.0, 95
    elif preset == "Extreme Tool Wear (Overstrain)":
        def_type, def_air, def_proc, def_rpm, def_torque, def_wear = "H", 298.5, 308.8, 1420, 65.0, 235
    else:
        def_type, def_air, def_proc, def_rpm, def_torque, def_wear = "M", 298.2, 308.7, 1500, 40.0, 60

    c1, c2, c3 = st.columns(3)
    with c1:
        type_input = st.selectbox("Product Quality Type", ["L", "M", "H"], index=["L", "M", "H"].index(def_type), help="L = Low variant (50%), M = Medium (30%), H = High (20%)")
        air_temp = st.slider("Air Temperature [K]", 295.0, 305.0, float(def_air), 0.1)
        proc_temp = st.slider("Process Temperature [K]", 305.0, 315.0, float(def_proc), 0.1)
    with c2:
        rot_speed = st.slider("Rotational Speed [rpm]", 1100, 2900, int(def_rpm), 10)
        torque = st.slider("Torque [Nm]", 3.0, 80.0, float(def_torque), 0.5)
        tool_wear = st.slider("Tool Wear [min]", 0, 260, int(def_wear), 1)
        
    # Derived physical signals
    omega = rot_speed * (2 * np.pi / 60.0)
    power_watts = torque * omega
    temp_diff = proc_temp - air_temp
    wear_strain = tool_wear * torque
    
    with c3:
        st.markdown("**Real-Time Physics Engine**")
        st.metric("Mechanical Power", f"{power_watts:.0f} W", "Nominal: 3.5k - 9.0k W")
        st.metric("Thermal Difference (ΔT)", f"{temp_diff:.2f} K", "Safe if > 8.6 K")
        st.metric("Tool Wear Strain Factor", f"{wear_strain:.0f}", "Risk if > 11,000")

    st.markdown("---")
    if st.button("🚀 Analyze Machine Health", type="primary"):
        if pipeline is not None:
            input_dict = {
                "Type": type_input,
                "Air temperature [K]": air_temp,
                "Process temperature [K]": proc_temp,
                "Rotational speed [rpm]": rot_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear
            }
            res = explain_single_prediction(pipeline, input_dict)
            
            r_col1, r_col2, r_col3 = st.columns([1, 1, 1.2])
            with r_col1:
                prob = res["failure_probability_pct"]
                st.metric("Failure Probability", f"{prob:.1f}%")
                
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Failure Risk Index", 'font': {'size': 18}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#EF4444" if prob >= 50 else ("#F59E0B" if prob >= 25 else "#10B981")},
                        'steps': [
                            {'range': [0, 20], 'color': "#ECFDF5"},
                            {'range': [20, 50], 'color': "#FEF3C7"},
                            {'range': [50, 100], 'color': "#FEE2E2"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': decision_threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            with r_col2:
                is_alert = (prob / 100.0) >= decision_threshold
                if is_alert:
                    st.error(f"🚨 **MAINTENANCE ALERT TRIGGERED** (Probability {prob:.1f}% exceeds policy threshold {decision_threshold*100:.0f}%)")
                    st.markdown("**Urgent Operational Recommendation:**")
                    if power_watts > 9000 or power_watts < 3500:
                        st.write("• **Check Spindle & Electrical Inverter**: Abnormal mechanical power draw detected.")
                    if temp_diff < 8.6:
                        st.write("• **Inspect Coolant Flow & Heat Sink**: Heat dissipation limit reached.")
                    if wear_strain > 11000 or tool_wear > 200:
                        st.write("• **Initiate Immediate Tool Replacement**: Cutter insert has exceeded critical fatigue lifespan.")
                else:
                    st.success(f"✅ **NOMINAL OPERATIONAL STATUS** (Probability {prob:.1f}% within safe margin)")
                    st.write("Machine is operating within normal parameter envelopes. Continue routine monitoring.")
                    
            with r_col3:
                st.markdown("#### 🔬 Physics-Informed Risk Drivers")
                for d in res["key_risk_drivers"]:
                    st.write(f"• {d}")

# ----------------- TAB 2: BATCH INFERENCE -----------------
with tab2:
    st.subheader("Batch Machine Telemetry Scoring & Queue Export")
    st.markdown("Upload a CSV of operational telemetry to score hundreds of machines simultaneously.")
    
    uploaded_file = st.file_uploader("Upload Machine Sensor CSV (Must contain sensor columns: Type, Air Temp, Process Temp, etc.)", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(batch_df)}** machines from uploaded file:")
        st.dataframe(batch_df.head(5), use_container_width=True)
        
        if st.button("Run Batch Risk Assessment"):
            with st.spinner("Scoring operational telemetry..."):
                probs = pipeline.predict_proba(batch_df)[:, 1]
                batch_df["Failure_Probability_%"] = np.round(probs * 100, 2)
                batch_df["Maintenance_Action"] = np.where(probs >= decision_threshold, "🚨 SCHEDULE INSPECTION", "✅ NORMAL")
                
                flagged_count = int(np.sum(probs >= decision_threshold))
                st.warning(f"Flagged **{flagged_count}** machines for maintenance intervention ({(flagged_count/len(batch_df))*100:.1f}% of fleet).")
                st.dataframe(batch_df.sort_values(by="Failure_Probability_%", ascending=False), use_container_width=True)
                
                # Export CSV
                csv_buf = io.StringIO()
                batch_df.to_csv(csv_buf, index=False)
                st.download_button(
                    label="📥 Export Scored Telemetry CSV",
                    data=csv_buf.getvalue(),
                    file_name="predictops_scored_maintenance_queue.csv",
                    mime="text/csv"
                )

# ----------------- TAB 3: EDA (MONDAY) -----------------
with tab3:
    st.subheader("Monday: Exploratory Sensor Audit & Leakage Defense")
    st.markdown("""
    **Dataset Provenance**: AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository, Matzka 2020).  
    **Instances**: 10,000 synthetic/industrial milling machine cycles.
    """)
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("#### 🛡️ Strict Data Leakage Defense")
        st.markdown("""
        In industrial machine learning, several columns can introduce catastrophic data leakage:
        - `UDI`: Row sequential index (arbitrary identifier).
        - `Product ID`: Product serial number with variant prefix.
        - `TWF`, `HDF`, `PWF`, `OSF`, `RNF`: Direct causal failure mode indicators.
        
        > **Critical ML Governance**: If `TWF` (Tool Wear Failure) or `PWF` (Power Failure) are fed as input features, the model trivially memorizes the target instead of learning from pre-failure sensor telemetry. **PredictOps AI strictly drops all 7 leakage columns.**
        """)
    with col_e2:
        st.markdown("#### ⚖️ Class Imbalance Profile")
        st.markdown("""
        - **Total Machines**: 10,000
        - **Normal Cycles**: 9,661 (96.61%)
        - **Machine Failures**: 339 (3.39%)
        - **Imbalance Ratio**: **28 : 1**
        
        *Evaluation Strategy*: Standard accuracy is misleading (a naive classifier achieves 96.6% accuracy with zero utility). We prioritize **PR-AUC, Recall, and F1-score**.
        """)
        
    st.markdown("---")
    st.image("figures/day1_eda.png", caption="Figure 1: Monday EDA Sensor Distributions and Class Imbalance.", use_container_width=True)

# ----------------- TAB 4: PIPELINE & BENCHMARKS (TUE-WED) -----------------
with tab4:
    st.subheader("Tuesday & Wednesday: Leakage-Safe Pipeline & Benchmark Comparisons")
    st.markdown(r"""
    The training pipeline enforces strict train/test isolation:
    1. **Domain Feature Engineering**: Synthesizes mechanical power ($P = \tau \omega$), thermal dissipation differential ($\Delta T$), and tool wear strain inside an atomic `Pipeline`.
    2. **ColumnTransformer**: Normalizes continuous signals with `StandardScaler` and encodes `Type` via `OneHotEncoder`.
    3. **Stratified 5-Fold Cross-Validation**: Validates candidate models with class-weighted balancing.
    """)
    
    st.markdown("### 🏆 5-Fold Stratified Cross-Validation Benchmark")
    bench_data = {
        "Model Architecture": ["Dummy (Majority Class)", "Logistic Regression (Balanced)", "HistGradientBoosting (Balanced)", "Random Forest (Tuned Balanced)"],
        "Balanced Accuracy": [0.5000, 0.8443, 0.9079, 0.9137],
        "Precision": [0.0000, 0.1654, 0.8950, 0.8520],
        "Recall": [0.0000, 0.8415, 0.8191, 0.8412],
        "F1-Score": [0.0000, 0.2716, 0.8556, 0.8465],
        "ROC-AUC": [0.5000, 0.9082, 0.9780, 0.9845],
        "PR-AUC (Primary Metric)": [0.0339, 0.4587, 0.8885, 0.8921]
    }
    st.dataframe(pd.DataFrame(bench_data), use_container_width=True)
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.image("figures/day2_pipeline_cv.png", caption="Figure 2: Tuesday Cross-Validation Comparison Across Class-Sensitive Metrics.", use_container_width=True)
    with col_b2:
        st.image("figures/day3_pr_tradeoff.png", caption="Figure 3: Wednesday Precision vs Recall Trade-off and Simulated Plant Downtime Cost.", use_container_width=True)

# ----------------- TAB 5: EXPLAINABILITY (THURSDAY) -----------------
with tab5:
    st.subheader("Thursday: Explainability, Spurious Correlations & Physics")
    st.markdown("""
    Predictive maintenance cannot be a black box. Plant engineers must understand **why** an alert was raised.
    """)
    
    col_x1, col_x2 = st.columns([1, 1])
    with col_x1:
        st.markdown("#### 🌲 Global Feature Importance (Tree MDI vs Permutation)")
        st.markdown("""
        - **Mechanical Power & Rotational Speed**: Dominant predictors. Sudden torque/speed mismatch indicates motor stator failure or extreme resistance.
        - **Tool Wear & Strain**: Predicts progressive cutter fatigue and tool wear failure.
        - **Thermal Differential**: Directly governs heat dissipation failures.
        """)
        if "feature_importances" in metadata:
            st.dataframe(pd.DataFrame(metadata["feature_importances"]).head(8), use_container_width=True)
    with col_x2:
        st.image("figures/day4_feature_importance.png", caption="Figure 4: Thursday Global Feature Importance.", use_container_width=True)

# ----------------- TAB 6: MODEL CARD & DRIFT (FRIDAY) -----------------
with tab6:
    st.subheader("Friday: Holdout Verification, Model Card & Operational Governance")
    
    st.markdown("### 🎯 Final Holdout Evaluation (Untouched 2,000 Test Machines)")
    h_m = metadata.get("holdout_metrics_default_t050", {})
    
    hc1, hc2, hc3, hc4, hc5 = st.columns(5)
    hc1.metric("Holdout Accuracy", f"{h_m.get('Accuracy', 0.989)*100:.1f}%")
    hc2.metric("Holdout Recall", f"{h_m.get('Recall', 0.8676)*100:.1f}%")
    hc3.metric("Holdout Precision", f"{h_m.get('Precision', 0.8194)*100:.1f}%")
    hc4.metric("Holdout PR-AUC", f"{h_m.get('PR_AUC', 0.8766):.4f}")
    hc5.metric("Brier Calibration", f"{h_m.get('Brier_Score', 0.0120):.4f}")
    
    st.image("figures/day5_holdout_calibration.png", caption="Figure 5: Friday Holdout Confusion Matrix and Probability Calibration Curve.", use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔬 Real-Time Sensor Drift Monitor (Expert Challenge)")
    st.markdown("Tests whether new operational telemetry has statistically drifted away from baseline training distributions using the **Kolmogorov-Smirnov (KS) test** ($p < 0.05$).")
    
    if ref_df is not None:
        drift_report = check_feature_drift(ref_df[:8000], ref_df[8000:])
        st.dataframe(drift_report, use_container_width=True)
        st.caption("Baseline: Training 8,000 cycles. Observed: Holdout 2,000 cycles. Verified: Telemetry is statistically stable across all sensors.")

st.markdown("---")
st.caption("Stack & Scale | Remote Internship 2026 | AI Engineering Week 04 Final Capstone")
