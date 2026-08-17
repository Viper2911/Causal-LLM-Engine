import pandas as pd
import numpy as np
import warnings
from dowhy import CausalModel

warnings.filterwarnings('ignore')

print("--- STARTING DIAGNOSTIC ---")

# 1. Generate clean, continuous dummy data
np.random.seed(42)
delayed = np.random.randint(0, 2, 100)
freight = np.random.uniform(10, 100, 100)
review = 5.0 - (1.5 * delayed) - (0.01 * freight) + np.random.normal(0, 0.5, 100)

df = pd.DataFrame({
    "delayed_shipping": delayed,
    "review_score": review,
    "freight_value": freight
})

# 2. Force strict types
df["delayed_shipping"] = df["delayed_shipping"].astype(bool)
df["review_score"] = df["review_score"].astype(float)
df["freight_value"] = df["freight_value"].astype(float)

print("Data built. Initializing DoWhy...")

# 3. NO TRY/EXCEPT BLOCKS - Let it crash so we see the trace
model = CausalModel(
    data=df,
    treatment="delayed_shipping",
    outcome="review_score",
    common_causes=["freight_value"]
)

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

print("Effect Identified. Running Estimate (Using Propensity Score Weighting)...")

# 4. We switch the math backend away from the buggy Linear Regression
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_weighting", 
    target_units="ate"
)

print(f"\nSUCCESS! ATE: {estimate.value}")