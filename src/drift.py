"""
Data Drift Detection Module for PredictOps AI (Optional Expert Challenge).
Uses Kolmogorov-Smirnov (KS) test to detect operational sensor drift.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

NUMERIC_SENSORS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

def check_feature_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Applies two-sample Kolmogorov-Smirnov test to detect distribution drift
    between baseline training data and newly observed sensor streams.
    """
    drift_records = []
    for col in NUMERIC_SENSORS:
        if col in baseline_df.columns and col in current_df.columns:
            s_base = baseline_df[col].dropna()
            s_curr = current_df[col].dropna()
            stat, p_val = ks_2samp(s_base, s_curr)
            is_drifted = bool(p_val < alpha)
            drift_records.append({
                "Sensor": col,
                "KS_Statistic": round(float(stat), 4),
                "P_Value": round(float(p_val), 5),
                "Status": "⚠️ DRIFT DETECTED" if is_drifted else "✅ STABLE",
                "Baseline_Mean": round(float(s_base.mean()), 2),
                "Current_Mean": round(float(s_curr.mean()), 2)
            })
    return pd.DataFrame(drift_records)
