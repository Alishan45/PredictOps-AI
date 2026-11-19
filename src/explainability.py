"""
Model Explainability and Local Sensor Attribution Module for PredictOps AI.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

def get_transformed_feature_names(pipeline) -> List[str]:
    """Retrieves feature names from the fitted ColumnTransformer inside the pipeline."""
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = []
    
    # Numeric features
    num_features = preprocessor.transformers_[0][2]
    feature_names.extend(num_features)
    
    # Categorical one-hot features
    cat_encoder = preprocessor.transformers_[1][1]
    cat_names = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2]).tolist()
    feature_names.extend(cat_names)
    
    return feature_names

def compute_global_feature_importance(pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> pd.DataFrame:
    """
    Computes both MDI (Gini / Tree Importance) and Permutation Importance
    to detect spurious feature dependencies.
    """
    feature_names = get_transformed_feature_names(pipeline)
    classifier = pipeline.named_steps["classifier"]
    
    # 1. Tree Mean Decrease in Impurity (MDI) on transformed features
    if hasattr(classifier, "feature_importances_"):
        tree_imp = classifier.feature_importances_
    else:
        tree_imp = np.zeros(len(feature_names))
        
    df_tree = pd.DataFrame({
        "Feature": feature_names,
        "Tree_MDI_Importance": np.round(tree_imp, 4)
    })

    # 2. Permutation Importance on input features
    perm_res = permutation_importance(
        pipeline, X_val, y_val,
        n_repeats=5,
        random_state=42,
        scoring="average_precision",
        n_jobs=-1
    )
    df_perm = pd.DataFrame({
        "Input_Feature": X_val.columns.tolist(),
        "Permutation_Importance_Mean": np.round(perm_res.importances_mean, 4),
        "Permutation_Importance_Std": np.round(perm_res.importances_std, 4)
    }).sort_values(by="Permutation_Importance_Mean", ascending=False).reset_index(drop=True)
    
    return df_tree.sort_values(by="Tree_MDI_Importance", ascending=False).reset_index(drop=True), df_perm

def explain_single_prediction(
    pipeline,
    raw_input_dict: Dict[str, Any],
    train_medians: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Provides human-interpretable explanation of failure risk for a single operational state.
    Calculates failure probability and pinpoints dominant physical risk drivers.
    """
    input_df = pd.DataFrame([raw_input_dict])
    
    # Predict failure probability
    prob = float(pipeline.predict_proba(input_df)[0, 1])
    
    # Evaluate physical thresholds based on domain knowledge:
    # 1. Power failure risk: Power = Torque * (Rot_speed * 2*pi/60)
    omega = float(raw_input_dict["Rotational speed [rpm]"]) * (2 * np.pi / 60.0)
    power_w = float(raw_input_dict["Torque [Nm]"]) * omega
    
    # 2. Thermal dissipation: Process Temp - Air Temp
    temp_diff = float(raw_input_dict["Process temperature [K]"]) - float(raw_input_dict["Air temperature [K]"])
    
    # 3. Tool wear strain
    wear = float(raw_input_dict["Tool wear [min]"])
    torque = float(raw_input_dict["Torque [Nm]"])
    strain = wear * torque
    
    drivers = []
    if power_w > 9000:
        drivers.append(f"Excessive Mechanical Power ({power_w:.0f} W > 9,000 W threshold)")
    elif power_w < 3500:
        drivers.append(f"Sub-critical Power Underload ({power_w:.0f} W < 3,500 W)")
        
    if temp_diff < 8.6 and wear > 150:
        drivers.append(f"Heat Dissipation Failure Risk (Temp Δ {temp_diff:.1f} K < 8.6 K with worn tool)")
        
    if strain > 11000:
        drivers.append(f"Severe Tool Overstrain (Strain factor {strain:.0f} > 11,000)")
    elif wear > 200:
        drivers.append(f"Tool Wear Near Replacement Limit ({wear:.0f} min > 200 min)")
        
    if not drivers:
        drivers.append("Operating within safe operational envelopes across all sensor telemetry")
        
    risk_level = "Critical" if prob >= 0.60 else ("High" if prob >= 0.35 else ("Moderate" if prob >= 0.15 else "Low"))
    
    return {
        "failure_probability_pct": round(prob * 100, 2),
        "risk_level": risk_level,
        "calculated_power_watts": round(power_w, 1),
        "calculated_temp_diff_kelvin": round(temp_diff, 2),
        "calculated_wear_strain": round(strain, 1),
        "key_risk_drivers": drivers
    }
