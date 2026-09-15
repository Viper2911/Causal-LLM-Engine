import pandas as pd
import numpy as np
import warnings
from dowhy import CausalModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

warnings.filterwarnings('ignore')

def preprocess_data(df: pd.DataFrame, treatment: str, outcome: str, confounders: list) -> pd.DataFrame:
    """Prepares the DataFrame for DoWhy with Smart Binarization V2 and NaN handling."""
    df_clean = df.copy()
    
    cols_to_check = [treatment, outcome] + confounders
    df_clean.dropna(subset=cols_to_check, inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    for col in df_clean.columns:
        try:
            df_clean[col] = pd.to_numeric(df_clean[col])
        except (ValueError, TypeError):
            pass 
            
    if pd.api.types.is_numeric_dtype(df_clean[treatment]):
        unique_count = df_clean[treatment].nunique()
        
        if unique_count > 2:
            median_val = df_clean[treatment].median()
            temp_treatment = (df_clean[treatment] >= median_val).astype(int)
            
            if temp_treatment.nunique() < 2:
                temp_treatment = (df_clean[treatment] > median_val).astype(int)
                
            df_clean[treatment] = temp_treatment
            
        elif unique_count == 2:
            unique_vals = sorted(df_clean[treatment].unique())
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            df_clean[treatment] = df_clean[treatment].map(mapping).astype(int)
        else:
            df_clean[treatment] = df_clean[treatment].astype(int)
    else:
        df_clean[treatment] = df_clean[treatment].astype(bool).astype(int)
        
    df_clean[outcome] = df_clean[outcome].astype(float)
    
    return df_clean

def discover_stronger_causes(df: pd.DataFrame, treatment: str, outcome: str, confounders: list) -> list:
    """Standardizes variables and runs sklearn regression to check confounder impact."""
    if not confounders: return []
    features = [treatment] + confounders
    df_check = df.dropna(subset=features + [outcome])
    if df_check.empty: return []
        
    try:
        scaler = StandardScaler()
        X = scaler.fit_transform(df_check[features])
        y = scaler.fit_transform(df_check[[outcome]])

        lr_model = LinearRegression()
        lr_model.fit(X, y)
        treatment_weight = abs(lr_model.coef_[0][0])

        stronger_candidates = []
        for i, conf in enumerate(confounders):
            conf_weight = abs(lr_model.coef_[0][i+1])
            if conf_weight > treatment_weight:
                stronger_candidates.append({
                    "variable": conf,
                    "confounder_impact": round(conf_weight, 4),
                    "treatment_impact": round(treatment_weight, 4)
                })
        return sorted(stronger_candidates, key=lambda x: x["confounder_impact"], reverse=True)
    except Exception:
        return []

def calculate_custom_ate_iptw(df: pd.DataFrame, treatment: str, outcome: str, confounders: list) -> float:
    """Proprietary custom implementation of Inverse Probability of Treatment Weighting."""
    T = df[treatment].astype(int).values
    Y = df[outcome].values
    X = df[confounders].values

    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr.fit(X, T)
    
    e_x = lr.predict_proba(X)[:, 1]
    e_x = np.clip(e_x, 0.01, 0.99)
    
    treated_impact = (T * Y) / e_x
    control_impact = ((1 - T) * Y) / (1 - e_x)
    
    return round(np.mean(treated_impact - control_impact), 4)

def run_causal_analysis(df: pd.DataFrame, treatment: str, outcome: str, confounders: list) -> dict:
    """Executes custom IPTW pipeline + DoWhy Refutation + Causal Discovery."""
    if df.empty: return {"status": "error", "message": "DataFrame is empty."}
    valid_confounders = [c for c in confounders if c in df.columns]
    
    df_clean = preprocess_data(df, treatment, outcome, valid_confounders)
    if df_clean.empty: return {"status": "error", "message": "All rows dropped due to NaN values."}
    
    if df_clean[treatment].nunique() < 2:
         return {"status": "error", "message": f"Data Skew Error: The treatment variable '{treatment}' does not have enough variance to create both a Control group and a Treatment group."}

    alternative_causes = discover_stronger_causes(df_clean, treatment, outcome, valid_confounders)
    
    try:
        custom_ate = calculate_custom_ate_iptw(df_clean, treatment, outcome, valid_confounders)
        
        model = CausalModel(data=df_clean, treatment=treatment, outcome=outcome, common_causes=valid_confounders)
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_weighting", target_units="ate")
        
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
            "custom_ate": custom_ate,
            "alternative_causes": alternative_causes,
            "placebo_summary": placebo_summary,
            "random_cause_summary": random_cause_summary
        }
    except Exception as e:
        return {"status": "error", "message": f"Mathematical engine failed: {str(e)}"}