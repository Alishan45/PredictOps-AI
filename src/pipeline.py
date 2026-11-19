"""
Robust Machine Learning Preprocessing and Pipeline Architecture for PredictOps AI.
Implements domain feature engineering, scaling, and encoding inside a leakage-free Scikit-Learn Pipeline.
"""

from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Synthesizes physical domain features:
      1. temp_difference [K] = Process temperature - Air temperature
      2. mechanical_power [W] = Torque [Nm] * Rotational speed [rpm] * (2 * pi / 60)
      3. wear_strain = Tool wear [min] * Torque [Nm]
    """
    def __init__(self):
        self.feature_names_out_: List[str] = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Work on a copy of DataFrame
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"])
        else:
            X_df = X.copy()

        # 1. Thermal Dissipation Differential
        temp_diff = X_df["Process temperature [K]"].astype(float) - X_df["Air temperature [K]"].astype(float)
        
        # 2. Rotational Mechanical Power in Watts (P = tau * omega)
        omega = X_df["Rotational speed [rpm]"].astype(float) * (2 * np.pi / 60.0)
        power_w = X_df["Torque [Nm]"].astype(float) * omega
        
        # 3. Tool Wear Strain Metric (Product of wear time and torque)
        wear_strain = X_df["Tool wear [min]"].astype(float) * X_df["Torque [Nm]"].astype(float)
        
        X_df["temp_difference"] = temp_diff
        X_df["mechanical_power"] = power_w
        X_df["wear_strain"] = wear_strain
        
        return X_df

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that scales continuous features
    and one-hot encodes the machine quality type.
    """
    numeric_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "temp_difference",
        "mechanical_power",
        "wear_strain"
    ]
    categorical_features = ["Type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor

def create_full_pipeline(classifier) -> Pipeline:
    """
    Chains domain feature engineering, column transformation,
    and the candidate classifier into an atomic pipeline.
    """
    pipeline = Pipeline(steps=[
        ("engineer", DomainFeatureEngineer()),
        ("preprocess", build_preprocessing_pipeline()),
        ("classifier", classifier)
    ])
    return pipeline
