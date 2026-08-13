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
