"""
Class-Sensitive Evaluation, Threshold Sensitivity, and Error Analysis for PredictOps AI.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_validate

def evaluate_model_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluates classifier across 5-fold Stratified Cross-Validation
    focusing on class-sensitive metrics.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision"
    }
    
    cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    return {
        "Accuracy": round(float(np.mean(cv_res["test_accuracy"])), 4),
        "Balanced_Accuracy": round(float(np.mean(cv_res["test_balanced_accuracy"])), 4),
        "Precision": round(float(np.mean(cv_res["test_precision"])), 4),
        "Recall": round(float(np.mean(cv_res["test_recall"])), 4),
        "F1_Score": round(float(np.mean(cv_res["test_f1"])), 4),
        "ROC_AUC": round(float(np.mean(cv_res["test_roc_auc"])), 4),
        "PR_AUC": round(float(np.mean(cv_res["test_pr_auc"])), 4)
    }

def evaluate_predictions(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics on predictions given a decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")
        
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = float("nan")
        
    brier = brier_score_loss(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "Model": model_name,
        "Threshold": round(threshold, 2),
        "Accuracy": round(float(acc), 4),
        "Balanced_Accuracy": round(float(bal_acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1_Score": round(float(f1), 4),
        "ROC_AUC": round(float(roc_auc), 4),
        "PR_AUC": round(float(pr_auc), 4),
        "Brier_Score": round(float(brier), 4),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    }

def analyze_threshold_spectrum(
    y_true: pd.Series,
    y_prob: np.ndarray,
    thresholds: List[float] = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
) -> pd.DataFrame:
    """Evaluates precision/recall trade-offs and operational costs across probability thresholds."""
    rows = []
    for t in thresholds:
        m = evaluate_predictions(y_true, y_prob, threshold=t)
        # Cost function: False Negative (breakdown) = $1,000, False Positive (inspection) = $50
        cost = (m["FN"] * 1000) + (m["FP"] * 50)
        rows.append({
            "Threshold": t,
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1_Score": m["F1_Score"],
            "TP (Caught)": m["TP"],
            "FP (Unnecessary Check)": m["FP"],
            "FN (Unplanned Breakdown)": m["FN"],
            "Estimated_Operational_Cost_$": cost
        })
    return pd.DataFrame(rows)
