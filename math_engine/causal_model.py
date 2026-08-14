import pandas as pd
import numpy as np
import warnings
from dowhy import CausalModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

def preprocess_data(df:pd.DataFrame, treatment:str, outcome:str)->pd.DataFrame:
    """Prepares the DataFrame for DoWhy."""
    df_clean=df.copy()
    df_clean.dropna(subset=[treatment,outcome],inplace=True)

    for col in df_clean.columns:
        if df_clean[col].dtype==bool:
            df_clean[col]=df_clean[col].astype(int)
    df_clean=df_clean.apply(pd.to_numeric,error='ignore')
    return df_clean

def discover_stronger_causes(df:pd.DataFrame, treatment:str, outcome:str, confounders:list)->list:
    """Standardized variables and runs regression to see if any confounder has a stronger mathematical impact on the outcome than the chosen treatment"""
    if not confounders:
        return []
    features=[treatment]+confounders
    df_check=df.dropna(subset=features+[outcome])
    if df_check.empty:
        return []
    try:
        scaler=StandardScaler()
        X=scaler.fit_transform(df_check[features])
        y=scaler.fit_transform(df_check[[outcome]])

        lr_model=LinearRegression()
        lr_model.fit(X,y)

        coefs=lr_model.coef_[0]
        treatment_weight=abs(coefs[0])

        stronger_candidates=[]
        for i, conf in enumerate(confounders):
            conf_weight=abs(coefs[i+1])
            if conf_weight>treatment_weight:
                stronger_candidates.append({
                    "variable":conf,
                    "confounder_impact":round(conf_weight,4),
                    "treatment_impact":round(treatment_weight,4)
                })
        return sorted(stronger_candidates,key=lambda x:x["confounder_impact"],reverse=True)
    except Exception as e:
        print(f"Discovery Enginer Warning: {e}")
        return []
def run_causal_analysis(df:pd.DataFrame, treatment:str, outcome:str, confounders:list)->dict:
    """Executes DoWhy pipeline + Causal Discovery."""
    if df.empty:
        return {"status":"error","message":"DataFrame is empty."}
    if treatment not in df.columns or outcome not in df.columns:
        return {"status":"error","message":"Treatment or Outcome missing from dataset."}
    df_clean=preprocess_data(df,treatment,outcome)
    valid_confounders=[c for c in confounders if c in df_clean.columns]

    alternative_causes=discover_stronger_causes(df_clean,treatment,outcome,valid_confounders)
    try:
        model=CausalModel(
            data=df_clean,
            treatment=treatment,
            outcome=outcome,
            common_causes=valid_confounders
        )
        identified_estimand=model.identify_effect(proceed_when_unidentifiable=True)
        estimate=model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            test_significance=True
        )
        placebo=model.refute_estimate(identified_estimand,estimate,method_name="placebo_treatment_refuter",num_simulations=10)
        random_cause=model.refute_estimate(identified_estimand,estimate,method_name="random_common_cause",num_simulations=10)
        
        return{
            "status":"success",
            "ate":round(float(estimate.value),4),
            "alternative_causes":alternative_causes,
            "estimate_summary":str(estimate),
            "placebo_summary":str(placebo),
            "random_cause_summary":str(random_cause)
        }
    except Exception as e:
        return {"status":"error","message":f"DoWhy calculation failed: {str(e)}"}

if __name__ == "__main__":
    print("Testing Causal Engine with Discovery...")
    dummy_data=pd.DataFrame({
        "delayed_shipping":[0,1,0,1,0,1,0,1,0,1],
        "review_score":[5,4,5,3,4,3,5,2,4,2],
        "freight_value":[10,80,15,90,12,85,10,95,14,88]
    })
    result=run_causal_analysis(dummy_data,"delayed_shipping","review_score",["freight_value"])
    print(f"ATE: {result['ate']}")
    if result['alternative_causes']:
        print("\n DISCOVERY ALERT: The engine found a stronger cause!")
        for alt in result['alternative_causes']:
            print(f"- '{alt['variable']}' has an impact score of {alt['confounder_impact']}, beating the treatment's score of {alt['treatment_impact']}.")