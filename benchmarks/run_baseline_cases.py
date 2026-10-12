"""Baseline benchmark on the same Olist cases: standard DoWhy + naive analyses.

For each canonical business question this script measures the *baseline*
pipeline that the project is compared against (standard DoWhy
propensity_score_weighting with default settings, as in the DoWhy paper and
the LLM+DoWhy workflows), on the exact same data as our engine:

  - plain median split (>= median, no validity guards - standard practice)
  - naive unadjusted difference in means
  - standard DoWhy backdoor.propensity_score_weighting
  - raw (unweighted) SMD and weighted SMD for the same confounders
  - placebo-treatment refuter on the DoWhy estimate (3 simulations, seed 42)
  - PEHE on the engine's synthetic ground-truth DGP (true ATE = 2.50) for
    naive difference, OLS and DoWhy PSW

Outputs benchmarks/results/baseline_cases.json.
"""

import json
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ai_engine.gemini_router import OFFLINE_CACHE  # noqa: E402
from database.db_setup import execute_query  # noqa: E402
from dowhy import CausalModel  # noqa: E402

OUT = REPO / "benchmarks" / "results" / "baseline_cases.json"
SEED = 42
KEYS = ["price", "freight", "photo", "description", "weight", "length", "installment"]


def standard_preprocess(df, treatment, outcome, confounders):
    """Plain median split with no validity guards, as a standard analyst
    pipeline would do. Kept separate from the engine's guarded binarization so
    the baseline stays a true baseline."""
    dfc = df.copy()
    dfc.dropna(subset=[treatment, outcome] + confounders, inplace=True)
    dfc.reset_index(drop=True, inplace=True)
    for col in dfc.columns:
        try:
            dfc[col] = pd.to_numeric(dfc[col])
        except (ValueError, TypeError):
            pass
    if pd.api.types.is_numeric_dtype(dfc[treatment]):
        n = dfc[treatment].nunique()
        if n > 2:
            dfc[treatment] = (dfc[treatment] >= dfc[treatment].median()).astype(int)
        elif n == 2:
            vals = sorted(dfc[treatment].unique())
            dfc[treatment] = dfc[treatment].map({vals[0]: 0, vals[1]: 1}).astype(int)
        else:
            dfc[treatment] = dfc[treatment].astype(int)
    else:
        dfc[treatment] = dfc[treatment].astype(bool).astype(int)
    dfc[outcome] = dfc[outcome].astype(float)
    return dfc


def compute_smd(df, treatment, confounders, propensity=None):
    """Average SMD across confounders; weighted if propensity scores given."""
    T = df[treatment].astype(int).values
    treated, control = T == 1, T == 0
    smds = []
    for conf in confounders:
        if conf not in df.columns:
            continue
        vals = pd.to_numeric(df[conf], errors="coerce").fillna(0).values.astype(float)
        var_t = np.var(vals[treated], ddof=1) if treated.sum() > 1 else 0.0
        var_c = np.var(vals[control], ddof=1) if control.sum() > 1 else 0.0
        pooled = math.sqrt(max((var_t + var_c) / 2.0, 1e-9))
        if propensity is None:
            diff = abs(float(np.mean(vals[treated]) - np.mean(vals[control])))
        else:
            e = np.clip(propensity, 0.05, 0.95)
            w = np.where(T == 1, 1.0 / e, 1.0 / (1.0 - e))
            wt, wc = w[treated], w[control]
            m_t = float(np.sum(wt * vals[treated]) / max(np.sum(wt), 1e-9))
            m_c = float(np.sum(wc * vals[control]) / max(np.sum(wc), 1e-9))
            diff = abs(m_t - m_c)
        smds.append(diff / pooled)
    return float(np.mean(smds)) if smds else 0.0


def dowhy_psw(df, treatment, outcome, confounders):
    model = CausalModel(data=df, treatment=treatment, outcome=outcome, common_causes=confounders)
    ident = model.identify_effect(proceed_when_unidentifiable=True)
    est = model.estimate_effect(
        ident, method_name="backdoor.propensity_score_weighting", target_units="ate"
    )
    return model, ident, est


def pehe_benchmark_baselines():
    np.random.seed(SEED)
    n = 1000
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    prob_t = 1 / (1 + np.exp(-(0.8 * x1 - 0.5 * x2)))
    t = np.random.binomial(1, prob_t)
    true_ate = 2.50
    y = true_ate * t + 1.8 * x1 + 2.2 * x2 + np.random.normal(0, 0.5, n)
    df = pd.DataFrame({"t": t, "y": y, "x1": x1, "x2": x2})

    naive = float(df[df.t == 1].y.mean() - df[df.t == 0].y.mean())
    ols = LinearRegression().fit(df[["t", "x1", "x2"]], df["y"])
    ols_ate = float(ols.coef_[0])
    try:
        _, _, est = dowhy_psw(df, "t", "y", ["x1", "x2"])
        psw_ate = float(est.value)
    except Exception as e:
        psw_ate = None
        print("PEHE DoWhy baseline failed:", str(e)[:120])

    out = {"true_ate": true_ate}
    for name, val in [("naive_diff", naive), ("unadjusted_ols", ols_ate), ("dowhy_psw", psw_ate)]:
        out[name] = None if val is None else {
            "estimated_ate": round(val, 4),
            "pehe": round(abs(val - true_ate), 4),
        }
    return out


def main():
    records = []
    for i, key in enumerate(KEYS, 1):
        plan = OFFLINE_CACHE[key]
        treat, out = plan["treatment"], plan["outcome"]
        conf = plan["confounders"]
        np.random.seed(SEED)

        df = execute_query(plan["sql_query"])
        rec = {"idx": i, "key": key, "treatment": treat, "outcome": out,
               "confounders": conf, "rows": len(df)}
        if df.empty:
            rec["status"] = "no_data"
            records.append(rec)
            continue

        clean = standard_preprocess(df, treat, out, conf)
        T = clean[treat].astype(int).values
        Y = clean[out].astype(float).values
        rec["n_treated"] = int(T.sum())
        rec["n_control"] = int(len(T) - T.sum())

        if T.sum() < 2 or (len(T) - T.sum()) < 2:
            rec["status"] = "degenerate_split"
            rec["error"] = f"groups {int(T.sum())} treated / {int(len(T)-T.sum())} control"
            records.append(rec)
            print(f"[{i}/7] {key:12s} degenerate_split", flush=True)
            continue

        rec["naive_ate"] = round(float(Y[T == 1].mean() - Y[T == 0].mean()), 4)

        X = clean[conf].values.astype(float)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, T)
        e_std = lr.predict_proba(X)[:, 1]

        rec["raw_smd"] = round(compute_smd(clean, treat, conf), 4)
        rec["weighted_smd_standard_psw"] = round(compute_smd(clean, treat, conf, e_std), 4)

        try:
            model, ident, est = dowhy_psw(clean, treat, out, conf)
            rec["dowhy_ate"] = round(float(est.value), 4)
            try:
                ref = model.refute_estimate(
                    ident, est, method_name="placebo_treatment_refuter",
                    num_simulations=3, random_state=SEED,
                )
                rec["placebo_new_effect"] = round(float(getattr(ref, "new_effect", 0.0)), 4)
            except Exception as e:
                rec["placebo_new_effect"] = None
                rec["placebo_error"] = str(e)[:120]
            rec["status"] = "success"
        except Exception as e:
            rec["status"] = "dowhy_error"
            rec["error"] = str(e)[:200]

        records.append(rec)
        print(f"[{i}/7] {key:12s} status={rec['status']} naive={rec.get('naive_ate')} "
              f"dowhy={rec.get('dowhy_ate')} rawSMD={rec.get('raw_smd')} "
              f"wSMD={rec.get('weighted_smd_standard_psw')}", flush=True)

    pehe = pehe_benchmark_baselines()
    payload = {"cases": records, "pehe_benchmark": pehe, "seed": SEED,
               "estimator": "standard DoWhy backdoor.propensity_score_weighting (default settings)"}
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print("\nPEHE synthetic benchmark:", json.dumps(pehe, indent=2))
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
