"""
Exploratory Data Analysis and Leakage-Safe Data Audit Module for PredictOps AI.
"""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TARGET_COL = "Machine failure"
LEAKAGE_COLS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
NUMERIC_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
CATEGORICAL_COLS = ["Type"]

def load_raw_dataset(csv_path: str = "data/ai4i2020.csv") -> pd.DataFrame:
    """Loads raw dataset and validates column integrity."""
    df = pd.read_csv(csv_path)
    # Strip potential UTF-8 BOM from column names
    df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
    return df

def audit_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Performs comprehensive data audit for class imbalance, missingness, and leakage."""
    missing_counts = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    
    target_counts = df[TARGET_COL].value_counts().to_dict()
    failure_rate = float(df[TARGET_COL].mean())
    
    audit_report = {
        "total_rows": len(df),
        "total_columns": df.shape[1],
        "missing_values_total": total_missing,
        "missing_per_column": missing_counts,
        "target_distribution": target_counts,
        "failure_rate_pct": round(failure_rate * 100, 2),
        "class_imbalance_ratio": f"{target_counts[0] // target_counts[1]}:1",
        "leakage_columns_identified": [c for c in LEAKAGE_COLS if c in df.columns],
        "numeric_feature_count": len(NUMERIC_COLS),
        "categorical_feature_count": len(CATEGORICAL_COLS)
    }
    return audit_report

def get_feature_distribution_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Computes summary statistics for continuous operational sensor features."""
    stats = []
    for col in NUMERIC_COLS:
        s = df[col]
        q25 = s.quantile(0.25)
        q75 = s.quantile(0.75)
        stats.append({
            "Feature": col,
            "Mean": round(float(s.mean()), 2),
            "Std": round(float(s.std()), 2),
            "Median": round(float(s.median()), 2),
            "IQR": round(float(q75 - q25), 2),
            "Min": round(float(s.min()), 2),
            "Max": round(float(s.max()), 2),
            "Skewness": round(float(s.skew()), 2)
        })
    return pd.DataFrame(stats)

def create_leakage_safe_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits data into train and untouched holdout test sets BEFORE any transformation.
    Explicitly drops all leakage columns (UDI, Product ID, specific failure modes).
    """
    # Drop target and leakage columns from feature matrix X
    drop_cols = [TARGET_COL] + [c for c in LEAKAGE_COLS if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test
