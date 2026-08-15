import pandas as pd
import numpy as np
import warnings
from dowhy import CausalModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

def preprocess_data(df: pd.DataFrame, treatment: str, outcome: str) -> pd.DataFrame:
    """Prepares the DataFrame for DoWhy."""
    df_clean = df.copy()
    
    df_clean.dropna(subset=[treatment, outcome], inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    for col in df_clean.columns:
        try:
            df_clean[col] = pd.to_numeric(df_clean[col])
        except (ValueError, TypeError):
            pass 
            
    df_clean[treatment] = df_clean[treatment].astype(float)
    df_clean[outcome] = df_clean[outcome].astype(float)
    
    return df_clean

def discover_stronger_causes(df: pd.DataFrame, treatment: str, outcome: str, confounders: list) -> list:
    """Standardizes variables and runs regression to see if any confounder has a stronger impact."""
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
        return {"status": "error", "message": "Treatment or Outcome missing from dataset."}
        
    df_clean = preprocess_data(df, treatment, outcome)
    valid_confounders = [c for c in confounders if c in df_clean.columns]

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
            method_name="backdoor.linear_regression",
            test_significance=False
        )
        
        try:
            placebo = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter", num_simulations=5)
            placebo_summary = str(placebo)
        except Exception:
            placebo_summary = "Placebo test skipped"
            
        try:
            random_cause = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause", num_simulations=5)
            random_cause_summary = str(random_cause)
        except Exception:
            random_cause_summary = "Random cause test skipped"
        
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

if __name__ == "__main__":
    print("Testing Causal Engine with 100 rows of robust simulated data...")
    
    np.random.seed(42)
    delayed = np.random.randint(0, 2, 100)
    freight = np.random.uniform(10, 100, 100)
    
    review = 5.0 - (1.5 * delayed) - (0.01 * freight) + np.random.normal(0, 0.5, 100)

    dummy_data = pd.DataFrame({
        "delayed_shipping": delayed,
        "review_score": review,
        "freight_value": freight
    })
    
    result = run_causal_analysis(dummy_data, "delayed_shipping", "review_score", ["freight_value"])
    
    if result["status"] == "success":
        print(f"SUCCESS! Engine ATE: {result['ate']}")
        if result['alternative_causes']:
            print("\n💡 DISCOVERY ALERT: The engine found a stronger cause!")
            for alt in result['alternative_causes']:
                print(f"- '{alt['variable']}' impact: {alt['confounder_impact']} vs Treatment impact: {alt['treatment_impact']}.")
    else:
        print(f"\nCRITICAL MATH ERROR: {result['message']}")