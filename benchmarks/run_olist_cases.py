"""Deterministic Olist benchmark: runs the engine's seven canonical business
questions on the live SQLite database (no Gemini API needed).

Treatment / outcome / confounder / SQL plans are taken from the offline
semantic cache in ai_engine/gemini_router.py, which is the exact same
parameter set the AI router returns for these questions.

Outputs benchmarks/results/olist_cases.json.
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ai_engine.gemini_router import OFFLINE_CACHE  # noqa: E402
from database.db_setup import execute_query  # noqa: E402
from math_engine.causal_model import preprocess_data, run_causal_analysis  # noqa: E402

OUT = REPO / "benchmarks" / "results" / "olist_cases.json"
SEED = 42

KEYS = ["price", "freight", "photo", "description", "weight", "length", "installment"]


def main():
    records = []
    for i, key in enumerate(KEYS, 1):
        plan = OFFLINE_CACHE[key]
        np.random.seed(SEED)

        df = execute_query(plan["sql_query"])
        rec = {
            "idx": i,
            "key": key,
            "hypothesis": plan["business_hypothesis"],
            "treatment": plan["treatment"],
            "outcome": plan["outcome"],
            "confounders": plan["confounders"],
            "rows": len(df),
        }
        if df.empty:
            rec["status"] = "no_data"
            records.append(rec)
            continue

        clean = preprocess_data(df, plan["treatment"], plan["outcome"], plan["confounders"])
        rec["n_clean"] = len(clean)
        rec["n_treated"] = int(clean[plan["treatment"]].sum())
        rec["n_control"] = int(len(clean) - rec["n_treated"])

        res = run_causal_analysis(df, plan["treatment"], plan["outcome"], plan["confounders"])
        rec["status"] = res.get("status")
        if res.get("status") == "success":
            rec["our_ate"] = res.get("custom_ate")
            rec["ci_95"] = res.get("ci_95")
            rec["p_value"] = res.get("p_value")
            rec["propensity_auc"] = res.get("propensity_auc")
            rec["refutation_pass_rate"] = res.get("refutation_pass_rate")
            rec["placebo_new_effect"] = res.get("placebo_new_effect")
            rec["random_cause_drift_pct"] = res.get("random_cause_drift_pct")
            rec["covariate_balance"] = res.get("covariate_balance")
            rec["pehe_benchmark"] = res.get("pehe_benchmark")
            rec["alternative_causes"] = res.get("alternative_causes")
        else:
            rec["error"] = res.get("message")

        records.append(rec)
        print(f"[{i}/{len(KEYS)}] {key:12s} status={rec['status']} rows={rec['rows']:,}", flush=True)

    OUT.write_text(json.dumps(records, indent=2, default=str))
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
