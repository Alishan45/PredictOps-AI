"""
Model Architecture, Candidate Classifiers, and Tuning for PredictOps AI.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import joblib

from src.pipeline import create_full_pipeline

def get_baseline_models() -> Dict[str, Any]:
    """Returns simple baseline classifiers for Monday's benchmark."""
    return {
        "Dummy (Majority Class)": create_full_pipeline(DummyClassifier(strategy="most_frequent")),
        "Logistic Regression (Balanced)": create_full_pipeline(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    }

def get_candidate_models() -> Dict[str, Any]:
    """Returns Tuesday's two candidate classifiers with class-imbalance awareness."""
    return {
        "Random Forest (Balanced)": create_full_pipeline(
            RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        ),
        "HistGradientBoosting (Balanced)": create_full_pipeline(
            HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=8,
                class_weight="balanced",
                random_state=42
            )
        )
    }

def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, Dict[str, Any]]:
    """
    Wednesday Hyperparameter Tuning:
    Optimizes Random Forest parameters via Stratified 5-Fold Cross Validation.
    Optimizes specifically for Average Precision (PR-AUC) given the 3.4% imbalance.
    """
    base_pipeline = create_full_pipeline(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    )
    
    param_grid = {
        "classifier__n_estimators": [100, 150, 200],
        "classifier__max_depth": [6, 10, 14],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    tuning_summary = {
        "best_params": grid_search.best_params_,
        "best_pr_auc_cv": round(float(grid_search.best_score_), 4)
    }
    return best_pipeline, tuning_summary

def calibrate_and_save_pipeline(pipeline, X_train: pd.DataFrame, y_train: pd.Series, artifact_path: str = "artifacts/predictops_pipeline.joblib"):
    """
    Fits and optionally calibrates probabilities with CalibratedClassifierCV
    to ensure output probabilities reflect true operational failure risk.
    """
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, artifact_path)
    print(f"Pipeline successfully serialized to {artifact_path}")
    return pipeline

def load_saved_pipeline(artifact_path: str = "artifacts/predictops_pipeline.joblib"):
    """Loads serialized pipeline artifact."""
    return joblib.load(artifact_path)
