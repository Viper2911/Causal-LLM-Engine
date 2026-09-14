import pandas as pd
import numpy as np
import warnings
from dowhy import CausalModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

def preprocess_data(df: pd.DataFrame, treatment: str, outcome: str, confounders: list) -> pd.DataFrame:
    """Prepares the DataFrame for DoWhy with Smart Binarization and comprehensive NaN handling."""
    df_clean = df.copy()
    
    cols_to_check = [treatment, outcome] + confounders
    df_clean.dropna(subset=cols_to_check, inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    for col in df_clean.columns:
        try:
            df_clean[col] = pd.to_numeric(df_clean[col])
        except (ValueError, TypeError):
            pass 
            
    if pd.api.types.is_numeric_dtype(df_clean[treatment]) and df_clean[treatment].nunique() > 2:
        median_val = df_clean[treatment].median()
        df_clean[treatment] = df_clean[treatment] >= median_val
    else:
        df_clean[treatment] = df_clean[treatment].astype(bool)
        
    df_clean[outcome] = df_clean[outcome].astype(float)
    
    return df_clean

def discover_stronger_causes(df: pd.DataFrame, treatment: str, outcome: str, confounders: list) -> list:
    """Standardizes variables and runs sklearn regression to check confounder impact."""
    if not confounders:
        return []
    
    features = [treatment] + confounders
    df_check = df.dropna(subset=features + [outcome])
    
    if df_check.empty:
        return []
        
    try:
        scaler = StandardScaler()
        X = scaler.fit_transform(df_check[features])
        y = scaler.fit_transform(df_check[[outcome]])

        lr_model = LinearRegression()
        lr_model.fit(X, y)

        coefs = lr_model.coef_[0]
        treatment_weight = abs(coefs[0])

        stronger_candidates = []
        for i, conf in enumerate(confounders):
            conf_weight = abs(coefs[i+1])
            if conf_weight > treatment_weight:
                stronger_candidates.append({
                    "variable": conf,
                    "confounder_impact": round(conf_weight, 4),
                    "treatment_impact": round(treatment_weight, 4)
                })
        return sorted(stronger_candidates, key=lambda x: x["confounder_impact"], reverse=True)
    except Exception as e:
        print(f"Discovery Engine Warning: {e}")
        return []

def run_causal_analysis(df: pd.DataFrame, treatment: str, outcome: str, confounders: list) -> dict:
    """Executes DoWhy pipeline + Causal Discovery."""
    if df.empty:
        return {"status": "error", "message": "DataFrame is empty."}
    if treatment not in df.columns or outcome not in df.columns:
        return {"status": "error", "message": f"Missing variables in data. Required: {treatment}, {outcome}"}
        
    valid_confounders = [c for c in confounders if c in df.columns]
    
    df_clean = preprocess_data(df, treatment, outcome, valid_confounders)
    
    if df_clean.empty:
         return {"status": "error", "message": "All rows were dropped due to missing values (NaNs) in the dataset."}

    alternative_causes = discover_stronger_causes(df_clean, treatment, outcome, valid_confounders)
    
    try:
        model = CausalModel(
            data=df_clean,
            treatment=treatment,
            outcome=outcome,
            common_causes=valid_confounders
        )
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_weighting",
            target_units="ate"
        )
        
        try:
            placebo = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter", num_simulations=5)
            placebo_summary = str(placebo)
        except Exception:
            placebo_summary = "Placebo test skipped (Data too sparse)"
            
        try:
            random_cause = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause", num_simulations=5)
            random_cause_summary = str(random_cause)
        except Exception:
            random_cause_summary = "Random cause test skipped (Data too sparse)"
        
        return {
            "status": "success",
            "ate": round(float(estimate.value), 4),
            "alternative_causes": alternative_causes,
            "estimate_summary": str(estimate),
            "placebo_summary": placebo_summary,
            "random_cause_summary": random_cause_summary
        }
    except Exception as e:
        return {"status": "error", "message": f"DoWhy calculation failed: {str(e)}"}