# Enterprise Causal AI Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LLM](https://img.shields.io/badge/Google%20GenAI-Gemini-orange.svg)](https://ai.google.dev/)
[![Causal Inference](https://img.shields.io/badge/Engine-Custom%20Hajek--IPTW%20%2B%20DoWhy-green.svg)](https://github.com/py-why/dowhy)
[![Frontend](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/Database-SQLite%20(Olist%20E--Commerce)-lightgrey.svg)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

An autonomous Causal AI system that translates natural-language business questions into multi-table SQL queries using semantic Retrieval-Augmented Generation (RAG), retrieves transactional data, and estimates the **Average Treatment Effect (ATE)** using Hajek-normalized Inverse Probability of Treatment Weighting (IPTW) and statistical refutation tests.

---

## Executive Summary

Traditional machine learning identifies associations in data, but associations do not necessarily establish causal relationships. Business decisions based only on correlations can therefore lead to misleading conclusions.

The Enterprise Causal AI Engine integrates natural-language processing, relational database querying, and statistical causal inference into a single workflow.

The system provides the following capabilities:

1. **Natural Language to Causal Parameters:** Converts business questions into structured treatment, outcome, and confounder variables.
2. **Multi-Table Relational SQL:** Retrieves relevant database schemas using semantic embeddings and executes SQL queries across relational tables.
3. **Mathematical Causal Estimation:** Estimates treatment effects using Hajek-normalized propensity weighting and bootstrap confidence intervals.
4. **Automated Causal Discovery:** Compares standardized variable impact scores to identify relationships that require further investigation.
5. **Statistical Stress Testing:** Evaluates model robustness using placebo treatment tests and synthetic common-cause injections.

---

## System Architecture

```text
User Business Question
         |
         v
+-------------------------------------------------------------+
| Tier 1: Intelligent AI and RAG Router                       |
|                                                             |
| - Schema embeddings using text-embedding-004                |
| - Cosine similarity for semantic table retrieval            |
| - Structured parameter extraction                           |
| - Treatment, outcome, confounders, and SQL generation       |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| Tier 2: Relational Database Execution Layer                 |
|                                                             |
| - SQLite relational database                                |
| - Multi-table SQL JOIN operations                           |
| - Data extraction and preprocessing                         |
| - Read-only SQL execution safeguards                        |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| Tier 3: Mathematical Causal Engine                          |
|                                                             |
| - Smart median continuous-treatment binarization            |
| - Logistic regression propensity-score estimation           |
| - Hajek-normalized IPTW ATE estimation                      |
| - Bootstrap confidence intervals and p-values               |
| - Standardized variable-impact analysis                     |
| - DoWhy placebo and synthetic-confounder refutation tests   |
+-----------------------------+-------------------------------+
                              |
                              v
+-------------------------------------------------------------+
| Tier 4: Streamlit Analytics Dashboard                       |
|                                                             |
| - ATE and confidence intervals                              |
| - Statistical significance                                  |
| - Covariate-balance diagnostics                             |
| - Propensity-model diagnostics                              |
| - Refutation results and analytical insights                |
+-----------------------------+-------------------------------+
                              |
                              v
                    BUSINESS INSIGHTS
```

---

## Mathematical Formulation

### 1. The Fundamental Problem of Causal Inference

In observational data, the same observation cannot simultaneously reveal its outcome under treatment and its outcome without treatment.

The observed outcome is represented as:

$$
Y_i = T_iY_i(1) + (1-T_i)Y_i(0)
$$

Where:

- \(Y_i\) is the observed outcome.
- \(T_i \in \{0,1\}\) is the treatment assignment.
- \(Y_i(1)\) is the potential outcome under treatment.
- \(Y_i(0)\) is the potential outcome without treatment.

Causal inference estimates the difference between these potential outcomes under the assumptions required by the selected method.

### 2. Custom Hajek-Normalized IPTW Estimator

The Inverse Probability of Treatment Weighting method estimates propensity scores to account for differences between treatment and control groups.

The propensity score is defined as:

$$
e(X_i)=P(T_i=1\mid X_i)
$$

Where \(X_i\) represents the selected confounders.

The treatment and control weights are:

$$
w_i^{(1)}=\frac{T_i}{e(X_i)}
$$

$$
w_i^{(0)}=\frac{1-T_i}{1-e(X_i)}
$$

The project uses a Hajek-normalized estimator:

$$
\widehat{ATE} =
\frac{\sum_{i=1}^{N}\frac{T_iY_i}{e(X_i)}}
{\sum_{i=1}^{N}\frac{T_i}{e(X_i)}}
-
\frac{\sum_{i=1}^{N}\frac{(1-T_i)Y_i}{1-e(X_i)}}
{\sum_{i=1}^{N}\frac{1-T_i}{1-e(X_i)}}
$$

Where:

- \(T_i\) represents treatment assignment.
- \(Y_i\) represents the observed outcome.
- \(e(X_i)\) is the propensity score estimated using logistic regression.
- The first term represents the weighted treatment-group mean.
- The second term represents the weighted control-group mean.

The difference between these weighted means estimates the Average Treatment Effect under the required causal assumptions.

The implementation clips propensity scores to the interval \([0.05,0.95]\) to limit extreme weights. This does not, by itself, guarantee that the positivity assumption is satisfied.

### 3. Covariate Balance (Standardized Mean Difference)

Covariate balance is the main diagnostic used to judge whether weighting removed confounding bias. For each confounder, the Standardized Mean Difference is computed before and after weighting:

$$
SMD = \frac{\left|\bar{X}_{1} - \bar{X}_{0}\right|}{\sqrt{\frac{s_1^2 + s_0^2}{2}}}
$$

Where \(\bar{X}_{1}, \bar{X}_{0}\) are the (weighted) confounder means in the treated and control groups, and \(s_1^2, s_0^2\) are the corresponding variances. A weighted SMD of 0.10 or below is the target for successful balance (Austin 2011, Stuart 2010). The dashboard reports the mean SMD across confounders and the percentage of bias removed:

$$
\text{Bias Reduction (\%)} = \frac{SMD_{raw} - SMD_{weighted}}{SMD_{raw}} \times 100
$$

### 4. Bootstrap Statistical Inference

The engine uses bootstrap resampling to estimate uncertainty in the treatment-effect estimate.

The documented configuration uses \(B=50\) bootstrap samples drawn with replacement.

The bootstrap standard error is calculated as:

$$
\widehat{SE}(\widehat{ATE}) =
\sqrt{
\frac{1}{B-1}
\sum_{b=1}^{B}
\left(
\widehat{ATE}^{*(b)}-\overline{ATE}^{*}
\right)^2
}
$$

The normal-approximation 95% confidence interval is:

$$
95\%\ CI =
\left[
\widehat{ATE}-1.96\widehat{SE},
\widehat{ATE}+1.96\widehat{SE}
\right]
$$

The two-sided p-value under a normal approximation is:

$$
p =
2\left[
1-\Phi\left(
\frac{|\widehat{ATE}|}{\widehat{SE}}
\right)
\right]
$$

Where \(\Phi\) represents the standard normal cumulative distribution function.

A p-value below 0.05 is commonly treated as statistically significant under the specified test. Statistical significance alone does not establish causality.

---

## Empirical Results Across 4 Real-World Test Cases

> **Note (engine hardening, 2026-10-12):** These runs were recorded before the engine was hardened with guarded binarization, DoWhy-aligned propensity clipping (`[0.05, 0.95]`) and a seeded `B = 50` bootstrap. The post-hardening measurements (7/7 valid estimates, 6/7 balanced within SMD <= 0.10, 90.8% lower PEHE than the naive baseline, 19x better balance on the installment question than the standard median-split pipeline) and the full comparison against the Microsoft Research baselines are documented in `RESULTS.docx` and reproducible from `benchmarks/results/`.

The results below come from live runs of the Streamlit dashboard on the Brazilian E-Commerce Public Dataset by Olist, which contains multiple relational tables covering orders, products, payments, shipping, and customer reviews.

| Run | Business Question | Treatment → Outcome (Controls) | Sample Size | Estimated ATE | 95% CI | p-value | Covariate Balance (SMD, Raw → Weighted) | Propensity AUC | Refutation Pass Rate |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Does a higher product price cause lower review scores, controlling for the shipping fee? | `price` → `review_score` (`freight_value`) | 112,372 | `+0.0789` | `[0.0502, 0.1076]` | `0.0001` | `0.5999 → 0.2443` (59.3% reduction) | `0.710` | `50%` |
| 2 | Does shipping delay cause lower review scores, controlling for freight price? | `freight_value` → `review_score` (`price`, `product_weight_g`) | 112,372 | `-0.1111` | `[-0.1827, -0.0395]` | `0.0023` | `0.5933 → 0.2293` (61.3% reduction) | `0.751` | `50%` |
| 3 | Does a higher count of product photos increase review scores when price is held constant? | `product_photos_qty` → `review_score` (`price`) | 110,774 | `+0.0745` | `[0.049, 0.100]` | `0.0001` | `0.076 → 0.004` (94.7% reduction) | `0.504` | `100%` |
| 4 | Does offering a discount on the product price increase the number of units sold, after controlling for product category and shipping fee? | `price` → `review_score` (`freight_value`) | 112,372 | `+0.0789` | `[-0.0127, 0.1705]` | `0.0912` | `0.5999 → 0.2443` (59.3% reduction) | `0.710` | `50%` |

### Key Findings per Run

1. **Price → Review Score:** The estimated effect is positive and the interval excludes zero, which is the opposite direction from the hypothesis that higher prices lower reviews. The weighted SMD for `freight_value` is 0.2443, above the 0.10 target, so residual confounding remains and the estimate should be treated as exploratory.
2. **Freight Value → Review Score:** The estimate is negative with a confidence interval that excludes zero (p = 0.0023). Balance is incomplete: weighted SMDs are 0.1674 for `price` and 0.2913 for `product_weight_g`, both above 0.10. The question asked about "shipping delay", but the router mapped it to `freight_value` (shipping cost), which is a different variable.
3. **Product Photos → Review Score:** This is the only run that meets the balance criterion (weighted SMD 0.004, all refutation tests passed). The propensity AUC of 0.504 means price barely predicts treatment, so there is little confounding to adjust for. The positive estimate is statistically significant.
4. **Discount → Units Sold (variable-mapping failure):** The question asked about units sold, but the router mapped it to the same `price` → `review_score` pair as Run 1 instead of a units-sold outcome. The result therefore does not answer the question asked. The interval includes zero (p = 0.0912), so the effect is only directional. The causal discovery module flagged `freight_value` as a stronger driver (impact score 0.0393 vs 0.011 for the requested treatment).

### Aggregate Diagnostics (Mean of the 4 Runs)

| Metric | Value | Reading |
| :--- | :---: | :--- |
| Mean raw SMD | `0.467` | Substantial confounding before weighting in 3 of 4 runs |
| Mean weighted SMD | `0.181` | Above the 0.10 target; only Run 3 is balanced |
| Mean bias reduction | `≈ 61%` | Weighting removed most, not all, of the imbalance |
| Mean propensity AUC | `0.669` | Runs 1, 2 and 4 fall in the 0.70 to 0.85 band; Run 3 is below it |
| Mean refutation pass rate | `62.5%` | Placebo ATE was near zero (0.002) in every run |
| Runs with p < 0.05 | `3 of 4` | Run 4 is directional only (p = 0.0912) |

### Interpretation Notes

- A weighted SMD above 0.10 (Runs 1, 2 and 4) means the Hajek-IPTW weighting did not fully balance the confounders, so those ATEs may still be biased.
- Runs 1 and 4 share the same point estimate (`+0.0789`) because they use the same variables and data, but their confidence intervals and p-values differ. This points to bootstrap variability with B = 50 resamples; increasing B would make the intervals more stable.
- A refutation pass rate of 50% means only half of the configured tests passed. It is not a probability that the estimate is correct.
- Statistical significance does not establish causality. It depends on treatment definition, confounder selection, overlap, and the absence of unmeasured confounding.

### Known Limitations

- **Variable mapping:** The RAG router can silently substitute variables when the requested concept (for example, units sold or shipping delay) has no matching column, as seen in Runs 2 and 4. Mapped treatment, outcome and confounders should be checked against the question before the ATE is trusted.
- **Residual imbalance:** Three of four runs finish with a weighted SMD above 0.10. Adding confounders such as product category, or trimming extreme propensity scores, may improve balance.
- **Bootstrap size:** B = 50 is small for stable confidence intervals.
- **Continuous treatments:** Treatments are binarized at the median, which discards dose-response information.

---

## Project Structure

```text
Causal-LLM-Engine/
|
├── ai_engine/
|   ├── __init__.py
|   ├── gemini_router.py
|   └── schema_embedder.py
|
├── database/
|   ├── __init__.py
|   ├── db_setup.py
|   └── olist.db
|
├── math_engine/
|   ├── __init__.py
|   └── causal_model.py
|
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

### File Descriptions

- `ai_engine/gemini_router.py`: Processes natural-language questions and extracts structured analysis parameters.
- `ai_engine/schema_embedder.py`: Creates schema embeddings and performs semantic table retrieval.
- `database/db_setup.py`: Initializes the SQLite database and configures database safeguards.
- `database/olist.db`: Stores the imported Olist relational data.
- `math_engine/causal_model.py`: Implements propensity weighting, ATE estimation, statistical evaluation, and configured refutation tests.
- `app.py`: Runs the Streamlit analytics dashboard.
- `requirements.txt`: Lists the project's Python dependencies.
- `.env`: Stores local environment configuration.
- `README.md`: Documents the project and its methodology.

The structure above should be adjusted if the actual repository contains different files or modules.

---

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Viper777/Causal-LLM-Engine.git
cd Causal-LLM-Engine
```

Replace the repository URL if your current GitHub repository uses a different address.

### 2. Set Up Virtual Environment

**Linux/macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

Ensure the environment variable name matches the one used by the application.

Do not commit your actual API key to GitHub.

### 5. Build Database (One-Time Setup)

```bash
python database/db_setup.py
```

Ensure the Olist source dataset is available at the location expected by the database initialization script.

### 6. Run the Application

```bash
streamlit run app.py
```

Open the local address displayed in the terminal, usually:

`http://localhost:8501`

---

## Academic Citation and References

```bibtex
@article{kiciman2023causal,
  title={Causal Reasoning and Large Language Models: Opening a New Frontier for Causality},
  author={Kiciman, Emre and Ness, Robert and Sharma, Amit and Tan, Chenhao},
  journal={arXiv preprint arXiv:2305.00050},
  year={2023}
}

@article{sharma2020dowhy,
  title={DoWhy: An End-to-End Library for Causal Inference},
  author={Sharma, Amit and Kiciman, Emre},
  journal={arXiv preprint arXiv:2011.04216},
  year={2020}
}
```

Additional resources:

- DoWhy Documentation: https://www.pywhy.org/dowhy/
- Olist Brazilian E-Commerce Dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- Google AI for Developers: https://ai.google.dev/
- Streamlit Documentation: https://docs.streamlit.io/