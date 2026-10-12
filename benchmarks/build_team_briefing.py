"""Builds TEAM_BRIEFING.docx: the complete briefing + faculty defense pack.

Everything a teammate needs to understand the project from scratch, explain the
causal-inference pipeline, demo it, and defend the comparison against the
Microsoft Research paper. Reads the measured results from benchmarks/results/.

Output: TEAM_BRIEFING.docx (repository root).
"""

import json
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
FIG_DIR = RESULTS / "figures"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
GREEN = RGBColor(0x00, 0x70, 0x3B)
RED = RGBColor(0xB0, 0x30, 0x2E)

olist = {r["key"]: r for r in json.loads((RESULTS / "olist_cases.json").read_text())}
before = {r["key"]: r for r in json.loads((RESULTS / "olist_cases_before_fix.json").read_text())}
base = json.loads((RESULTS / "baseline_cases.json").read_text())
base_cases = {r["key"]: r for r in base["cases"]}
pehe = base["pehe_benchmark"]

OURS_PEHE = olist["photo"]["pehe_benchmark"]["pehe_score"]
TRUE_ATE = olist["photo"]["pehe_benchmark"]["true_ate"]
ORDER = ["price", "freight", "photo", "description", "weight", "length", "installment"]
NAMES = {
    "price": "Product price -> review score",
    "freight": "Freight cost -> review score",
    "photo": "Product photos -> review score",
    "description": "Description length -> review score",
    "weight": "Product weight -> freight cost",
    "length": "Product length -> freight cost",
    "installment": "Installments -> payment value",
}
CONTROLS = {
    "price": "freight_value",
    "freight": "price, product_weight_g",
    "photo": "price",
    "description": "price, product_photos_qty",
    "weight": "price",
    "length": "product_weight_g",
    "installment": "payment_sequential",
}
n_balanced = sum(1 for k in ORDER if olist[k]["covariate_balance"]["weighted_smd"] <= 0.10)
mean_bias = sum(olist[k]["covariate_balance"]["bias_reduction_pct"] for k in ORDER) / len(ORDER)
sig = [k for k in ORDER if olist[k]["p_value"] < 0.05]
ours_mean_wsmd = sum(olist[k]["covariate_balance"]["weighted_smd"] for k in ORDER) / len(ORDER)
base_mean_wsmd = sum(r["weighted_smd_standard_psw"] for r in base["cases"]
                     if r.get("weighted_smd_standard_psw") is not None and r["status"] == "success") / 6


def shade(cell, rgb_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.makeelement(qn("w:shd"), {qn("w:val"): "clear", qn("w:fill"): rgb_hex})
    tcPr.append(shd)


def table(doc, rows, widths=None, header=True, highlight=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Light Grid Accent 1"
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = str(val)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
            if header and i == 0:
                shade(cell, "1F4E79")
                for p in cell.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for r in p.runs:
                        r.font.bold = True
                        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    if highlight:
        for green in highlight:
            shade(t.cell(green[0], green[1]), "E2EFDA")
    if widths:
        for j, w in enumerate(widths):
            for i in range(len(rows)):
                t.cell(i, j).width = Inches(w)
    return t


def h(doc, text, level=1):
    doc.add_heading(text, level=level)


def p(doc, text, bold_=False, italic=False, size=10.5, color=None, align=None, style=None):
    par = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    run = par.add_run(text)
    run.bold = bold_
    run.italic = italic
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color
    if align:
        par.alignment = align
    return par


def bullet(doc, text, bold_prefix=None, size=10):
    par = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        r = par.add_run(bold_prefix)
        r.bold = True
        r.font.size = Pt(size)
    r = par.add_run(text)
    r.font.size = Pt(size)
    return par


def numbered(doc, text, size=10):
    par = doc.add_paragraph(style="List Number")
    r = par.add_run(text)
    r.font.size = Pt(size)
    return par


def code(doc, text):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    par.paragraph_format.space_after = Pt(2)
    return par


def main():
    doc = Document()
    doc.core_properties.title = "Enterprise Causal AI Engine - Team Briefing & Faculty Defense Pack"
    doc.core_properties.author = "Enterprise Causal AI Engine"
    for section in doc.sections:
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)

    # ================= COVER =================
    p(doc, "Enterprise Causal AI Engine", bold_=True, size=28, color=ACCENT, align=WD_ALIGN_PARAGRAPH.CENTER)
    p(doc, "Team Briefing & Faculty Defense Pack", bold_=True, size=18, align=WD_ALIGN_PARAGRAPH.CENTER)
    p(doc, "Everything you need to know and say tomorrow - from first principles to the word-for-word demo script",
      italic=True, size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER)
    p(doc, f"Document date: {date.today().isoformat()}   |   Evidence: benchmarks/results/*.json   |   Companion: RESULTS.docx",
      size=9.5, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    p(doc, "")
    p(doc, "READ THIS FIRST: Sections 1-4 explain the project from scratch. Section 5 walks through a real run "
           "number by number. Sections 6-8 cover the comparison against the Microsoft Research paper. Section 9 is "
           "the exact demo script. Section 10 is the Q&A rehearsal. If you only have 15 minutes, read Section 0, "
           "Section 5, Section 8 and Section 9.", size=10.5, italic=True)

    # ================= 0. SUMMARY =================
    h(doc, "0. The 60-Second Summary (memorize this)", 1)
    bullet(doc, "It is an end-to-end causal AI engine: you ask a business question in plain English, and it returns "
                "an Average Treatment Effect (ATE) with a 95% confidence interval, a p-value, confounding-bias "
                "diagnostics (SMD), refutation test results and causal-discovery alerts.")
    bullet(doc, "The language model never computes a single number. It only translates English into SQL and into "
                "variables (treatment, outcome, confounders). Every statistic is deterministic Python code - "
                "auditable and reproducible.")
    bullet(doc, "It runs on 100,000+ real Olist e-commerce rows: 7/7 canonical questions produce valid estimates, "
                f"{n_balanced}/7 reach the gold-standard balance target (SMD <= 0.10), removing {mean_bias:.1f}% of "
                "measured confounding bias on average (up to 99.4%).")
    bullet(doc, f"Accuracy against known ground truth: PEHE = {OURS_PEHE:.4f} vs the naive baseline's "
                f"{pehe['naive_diff']['pehe']:.4f} - a 90.8% error reduction - and it matches the gold-standard "
                "DoWhy estimator exactly on every shared case (verification, not luck).")
    bullet(doc, "Where the standard pipeline breaks, ours holds: it silently reported a spurious +59.79 from a "
                "2-row control group on the installments question (SMD 1.816); our guarded engine answers it with "
                "SMD 0.095 (19x better).")
    bullet(doc, "The comparator paper (Microsoft Research, Kiciman et al. 2023/2024) measures how well LLMs "
                "*reason about causality in text*. It performs no effect estimation from data - the paper itself "
                "says \"LLMs ignore the actual data\". We are the data-grounded estimation layer that completes "
                "that pipeline, and we beat the standard LLM+DoWhy pipeline on the metrics that matter.")

    doc.add_page_break()

    # ================= 1. PROBLEM =================
    h(doc, "1. The Problem and the Idea (from scratch)", 1)
    h(doc, "1.1 What a causal question is", 2)
    p(doc, "A causal question is not \"what is associated with what?\" but \"what happens to Y if I intervene on X?\" "
           "For example: \"If I make shipping cheaper, will review scores improve?\" or \"If products had more "
           "photos, would customers rate them higher?\" Businesses constantly ask these questions, and they cannot "
           "be answered by a prediction model alone.")
    h(doc, "1.2 Why normal machine learning is not enough", 2)
    p(doc, "Standard ML learns associations: patterns like P(Y | X). Association is not causation. The classic "
           "example: ice-cream sales correlate with shark attacks - but banning ice cream will not reduce shark "
           "attacks; both are driven by summer weather. In business terms: expensive products may correlate with "
           "lower ratings, but the real driver might be something else that travels with price (shipping cost, "
           "product category, seller quality). A predictor that ignores this will recommend the wrong action.")
    p(doc, "The clean way to answer causal questions is a randomized experiment (A/B test). But you cannot "
           "randomize the number of product photos a listing had over the last year, and you cannot re-run "
           "100,000 historical orders with different shipping costs. So we must use observational causal "
           "inference - careful statistics that reconstruct a fair comparison from data we already have.")
    h(doc, "1.3 Why this is hard today", 2)
    p(doc, "Professional causal inference tooling (DoWhy, EconML) is powerful but gatekept: you must write your own "
           "SQL, know which variables to control for, build a causal graph, choose an estimator, and validate it. "
           "That is PhD-level work for every single business question.")
    h(doc, "1.4 Our idea", 2)
    p(doc, "Split the job along the line where machines are reliable:")
    bullet(doc, "The LLM does language work only - it finds the right tables, writes SQL, and identifies the "
                "treatment, outcome and confounders. It never estimates anything.", bold_prefix="LLM = translator. ")
    bullet(doc, "Deterministic code does the causal statistics - propensity scores, weighting, "
                "bias diagnostics, uncertainty and stress tests. Same input, same output, every time.",
           bold_prefix="Statistics = engine. ")
    bullet(doc, "Every number carries its own report card: balance before/after weighting, confidence intervals, "
                "refutation tests and discovery alerts. If the data cannot support an estimate, the engine says so.",
           bold_prefix="Trust = diagnostics. ")

    # ================= 2. INPUT/OUTPUT =================
    h(doc, "2. What the System Takes In and Gives Back", 1)
    p(doc, "Input (one sentence, plain English):", bold_=True)
    code(doc, "\"Does having more product photos cause higher review scores, controlling for price?\"")
    p(doc, "Output (computed automatically on live data):", bold_=True)
    rows = [
        ["Output", "Example (photos -> reviews)", "What it means"],
        ["Treatment / outcome / confounders", "product_photos_qty / review_score / [price]", "The causal question formalized"],
        ["Generated SQL", "3-table JOIN over 110,774 rows", "How data was actually pulled"],
        ["ATE (Average Treatment Effect)", "+0.0745 stars", "Estimated causal effect of the treatment"],
        ["95% confidence interval", "[0.030, 0.119]", "Uncertainty range (seeded bootstrap, B=50)"],
        ["p-value", "0.0011", "Probability the effect is a chance artifact"],
        ["Covariate balance (SMD)", "0.076 -> 0.004 after weighting", "How much confounding bias was removed"],
        ["Propensity AUC", "0.504", "How predictable treatment was from controls (positivity check)"],
        ["Refutation pass rate", "100%", "Placebo + random-cause stress tests"],
        ["Causal discovery alerts", "none / ranked strongest drivers", "Warns if a control outranks the treatment"],
    ]
    table(doc, rows, widths=[1.9, 2.4, 2.6], highlight=[(6, 2)])

    doc.add_page_break()

    # ================= 3. ARCHITECTURE =================
    h(doc, "3. The System, Tier by Tier", 1)
    code(doc, "English question -> [1] RAG + LLM router -> [2] SQLite SQL -> [3] Causal math engine -> [4] Streamlit dashboard")
    h(doc, "3.1 Tier 1a - Schema retrieval (ai_engine/schema_embedder.py)", 2)
    bullet(doc, "The database has 5 relevant tables. Each table's schema is embedded into a vector using Google "
                "text-embedding-004 (with a deterministic local fallback if the API is unavailable).")
    bullet(doc, "The user's question is embedded the same way, and the 3 most similar tables are selected by "
                "cosine similarity:  sim = (q . t) / (||q|| ||t||). This is classic RAG: give the LLM only the "
                "schema it needs.")
    h(doc, "3.2 Tier 1b - The LLM router (ai_engine/gemini_router.py)", 2)
    bullet(doc, "Gemini (model id gemini-3.8-flash) receives the retrieved schema plus the question and must return "
                "strict JSON: sql_query, treatment, outcome, confounders, business_hypothesis.")
    bullet(doc, "The prompt explicitly frames the model as an expert data scientist; retries handle transient "
                "errors; a deterministic offline cache answers the canonical demo questions even with no network.")
    bullet(doc, "Key design rule: JSON out, numbers never. The model's job ends at naming variables and writing SQL.")
    h(doc, "3.3 Tier 2 - The database layer (database/db_setup.py)", 2)
    bullet(doc, "SQLite database (database/olist.db), built from the Brazilian E-Commerce (Olist) dataset: "
                "orders 99,441 rows; order_reviews 99,224; order_items 112,650; products 32,951; "
                "order_payments 103,886.")
    bullet(doc, "A read-only guard allows only SELECT / WITH queries - the generated SQL can never modify data.")
    bullet(doc, "Queries are real multi-table JOINs (e.g. reviews JOIN items JOIN products), so the analysis runs "
                "on live transactional records, not a spreadsheet.")
    h(doc, "3.4 Tier 3 - The causal math engine (math_engine/causal_model.py)", 2)
    p(doc, "This is the scientific core. Its sub-steps, in order of execution:")
    rows = [
        ["Step", "Function", "What it does"],
        ["1", "preprocess_data", "Cleans NaN, coerces types, binarizes continuous treatments with guards"],
        ["2", "discover_stronger_causes", "Standardized regression: does any control outrank the treatment?"],
        ["3", "calculate_custom_ate_iptw", "Propensity model -> Hajek IPTW ATE + bootstrap CI/p-value"],
        ["4", "calculate_covariate_balance", "Raw vs weighted SMD per confounder; % bias removed"],
        ["5", "calculate_pehe_benchmark", "Synthetic ground-truth test: is the estimator accurate?"],
        ["6", "DoWhy refuters", "Placebo treatment + random common cause stress tests"],
        ["7", "run_causal_analysis", "Orchestrates everything and returns the report dict"],
    ]
    table(doc, rows, widths=[0.5, 2.3, 4.1])
    h(doc, "3.5 Tier 4 - The dashboard (app.py)", 2)
    bullet(doc, "Streamlit UI: metrics cards for ATE, SMD (main metric), propensity AUC, refutation pass rate and "
                "p-value; a raw data preview; the generated SQL; the live balance table; discovery alerts; and the "
                "raw DoWhy refuter logs.")

    doc.add_page_break()

    # ================= 4. THE MATH =================
    h(doc, "4. Causal Inference, Explained from Scratch", 1)
    h(doc, "4.1 Potential outcomes and the ATE", 2)
    p(doc, "Every order/item has two potential outcomes: Y(1) if treated (e.g. many photos), Y(0) if not. The "
           "individual causal effect is Y(1) - Y(0). We never see both for the same unit - this is the fundamental "
           "problem of causal inference. What we can estimate is the Average Treatment Effect over the population:")
    code(doc, "ATE = E[ Y(1) - Y(0) ]")
    p(doc, "Because we only observe Y = T*Y(1) + (1-T)*Y(0), a naive mean difference between treated and control "
           "groups is biased whenever the groups differ in other ways. That bias is confounding.")
    h(doc, "4.2 Why accuracy and F1-score are meaningless here", 2)
    p(doc, "Accuracy/F1 need labels of the counterfactual outcome - which do not exist. A model can predict review "
           "scores with 99% accuracy and still tell you nothing about what would happen if you changed shipping. "
           "Causal evaluation instead uses: estimation error against known ground truth (PEHE), bias diagnostics "
           "(SMD), uncertainty (CI/p), and stress tests (refutations).")
    h(doc, "4.3 Confounders and why we weight", 2)
    p(doc, "A confounder is a variable that influences both treatment and outcome (price influences how many "
           "photos a listing has and also review scores). If treated and control groups differ on confounders, "
           "the comparison is unfair. The fix: re-weight each group so that, after weighting, both look as if "
           "they came from the same population.")
    h(doc, "4.4 Propensity scores", 2)
    p(doc, "Fit a logistic regression predicting treatment from the confounders. The predicted probability is the "
           "propensity score e(X) = P(T=1 | X). The AUC of this model tells us how separable the groups are:")
    bullet(doc, "AUC ~ 0.5 - treatment is essentially random given the controls (little confounding). ")
    bullet(doc, "AUC 0.7-0.85 - controls genuinely predict treatment (adjustment matters); ")
    bullet(doc, "AUC near 1.0 - groups barely overlap (positivity danger: estimates become fragile).")
    h(doc, "4.5 IPTW and the Hajek estimator", 2)
    p(doc, "Inverse Probability of Treatment Weighting makes each unit represent the population: treated units get "
           "weight 1/e(X), control units 1/(1-e(X)). Horvitz-Thompson averages these weighted outcomes. The "
           "Hajek (self-normalized) version divides by the sum of weights, keeping weights summing to 1 and "
           "taming extreme values:")
    code(doc, "ATE_Hajek = SUM(T*Y/e) / SUM(T/e)  -  SUM((1-T)*Y/(1-e)) / SUM((1-T)/(1-e))")
    p(doc, "Propensity scores are clipped to [0.05, 0.95] - the same interval DoWhy uses - so no single unit can "
           "dominate the estimate. This is deliberately aligned with the gold-standard implementation.")
    h(doc, "4.6 SMD - the metric that says how much our project is right", 2)
    p(doc, "The Standardized Mean Difference measures how different treated and control groups are on each "
           "confounder, in standard-deviation units:")
    code(doc, "SMD = | mean_T(X) - mean_C(X) | / sqrt( (var_T + var_C) / 2 )")
    p(doc, "SMD is computed twice per confounder - before weighting (raw) and after weighting - and the engine "
           "reports the percentage of bias removed = (raw - weighted)/raw * 100. The accepted target is SMD <= "
           "0.10 (Austin 2011; Stuart 2010). This metric is evaluated directly on real data, which is why it is "
           "the engine's headline trust metric: low weighted SMD means the comparison is now fair.")
    h(doc, "4.7 Uncertainty: bootstrap, CI and p-value", 2)
    p(doc, "The engine draws B = 50 bootstrap resamples (seeded, seed 42) and re-estimates the ATE each time. "
           "The spread of those estimates gives the standard error; the 95% CI is estimate +/- 1.96*SE; the "
           "p-value is 2*(1 - Phi(|ATE|/SE)). Seeding makes every number reproducible.")
    h(doc, "4.8 Binarization and its guards", 2)
    p(doc, "Our ATE is a contrast between two groups, but most business variables are continuous (freight value, "
           "price, weight). Standard practice is a median split. A median split can fail badly: for installments, "
           "the median is 1 and almost every order has >= 1 installment - a naive split creates a control group "
           "of 2 rows and a meaningless estimate. Our guarded binarization:")
    numbered(doc, "try the normal split (>= median);")
    numbered(doc, "if either group is smaller than 30 rows or 1% of the data, use the strict split (> median) - "
                  "for installments this cleanly separates '1 installment' from '2 or more';")
    numbered(doc, "if neither split is valid, refuse to estimate and say why. No silent garbage.")
    h(doc, "4.9 Refutation tests - trying to break our own result", 2)
    bullet(doc, "Placebo treatment: replace the real treatment with a random one; the estimated effect should "
                "collapse to ~0. If it does not, the pipeline is detecting noise as signal. Pass rule: "
                "|placebo effect| < max(0.05, 15% of the real ATE).")
    bullet(doc, "Random common cause: add a random noise confounder and re-estimate; the answer should barely "
                "move (drift <= 10%). If it swings, the estimate is fragile.")
    bullet(doc, "These are implemented through DoWhy on a 10,000-row representative sample and run on every "
                "question.")
    h(doc, "4.10 Causal discovery - catching the wrong question", 2)
    p(doc, "A standardized linear regression ranks the empirical leverage of the treatment against each control. "
           "If a control has a stronger standardized coefficient than the requested treatment, the engine raises "
           "a discovery alert. Example: for 'description length -> reviews' it found product photos carry 1.7x "
           "the leverage; for 'product length -> freight' it found weight dominates by ~555x - so the length "
           "coefficient should not be over-interpreted.")
    h(doc, "4.11 The PEHE ground-truth test (how we prove accuracy)", 2)
    p(doc, "Simulate data with a known answer: two confounders x1, x2 generate treatment assignment and the "
           "outcome y = 2.50*T + 1.8*x1 + 2.2*x2 + noise. The true effect is exactly 2.50. We then ask each "
           "method to recover it and report PEHE = |estimate - 2.50|. Our engine: 2.4851 (PEHE 0.0149). Naive "
           "difference in means: 2.6614 (PEHE 0.1614). This is the cleanest proof that the estimator is accurate "
           "and that adjustment matters.")

    doc.add_page_break()

    # ================= 5. WORKED EXAMPLE =================
    h(doc, "5. A Real Run, Number by Number (use this in the demo)", 1)
    p(doc, "Question: \"Does having more product photos cause higher customer review scores, controlling for "
           "the product's price?\"", bold_=True)
    rows = [
        ["Step", "What happened", "Number"],
        ["1. RAG", "Embedded the question, retrieved the top-3 most similar table schemas", "products, order_reviews, order_items"],
        ["2. LLM router", "Returned strict JSON with variables and SQL", "T=product_photos_qty, Y=review_score, Z=[price]"],
        ["3. SQL", "Executed a 3-table JOIN on live SQLite data", "110,774 rows"],
        ["4. Binarization", "Median split with guard (photos > median vs <= median)", "54,831 treated / 55,943 control"],
        ["5. Propensity", "Logistic model of treatment on price", "AUC 0.504 (little confounding)"],
        ["6. Balance", "SMD computed before and after Hajek weighting", "0.076 -> 0.004 (94.7% bias removed)"],
        ["7. ATE", "Hajek IPTW estimate", "+0.0745 stars"],
        ["8. Uncertainty", "50-resample seeded bootstrap", "95% CI [0.0299, 0.1191], p = 0.0011"],
        ["9. Refuters", "Placebo + random-cause stress tests", "2/2 passed (100%)"],
        ["10. Discovery", "Standardized leverage ranking", "no stronger confounder found"],
    ]
    table(doc, rows, widths=[1.2, 3.1, 2.5], highlight=[(6, 2), (7, 2), (9, 2)])
    p(doc, "How to read it: after weighting, treated and control groups are almost identical on price "
           "(SMD 0.004, far below the 0.10 target), the effect is positive and statistically meaningful "
           "(95% CI excludes zero, p = 0.0011), and both stress tests pass. Conclusion: richer listing imagery "
           "causally lifts ratings by about +0.075 stars per 'more photos' vs 'fewer photos' contrast.", size=10)
    h(doc, "5.1 The kicker example: installments and the 2-row trap", 2)
    p(doc, "Same pipeline on payments data (103,886 rows): the treatment 'payment_installments' has median 1 - "
           "almost every order pays in at least one installment. A standard median split puts 103,884 orders in "
           "one group and 2 orders in the other. The standard DoWhy pipeline accepts this and silently reports "
           "an ATE of +59.79 with absurd imbalance (SMD 1.816). Our guarded split instead compares 1 installment "
           "vs 2+ (52,548 vs 51,338 orders), removes 62.2% of measured imbalance (SMD 0.251 -> 0.095) and "
           "reports +81.93 with a 95% CI of [75.65, 88.21] and p = 0.0001. Same data, same question - one "
           "pipeline quietly produces an artifact, ours produces a defensible estimate.")

    # ================= 6. THE PAPER =================
    h(doc, "6. The Comparator Paper, Explained", 1)
    p(doc, "Paper: Kiciman, Ness, Sharma & Tan (Microsoft Research), \"Causal Reasoning and Large Language "
           "Models: Opening a New Frontier for Causality\" (arXiv:2305.00050). It is the landmark behavioural "
           "study of LLM causal reasoning, and it is already cited in this project's README. Crucially, its "
           "authors are also the creators of DoWhy (Sharma & Kiciman, 2020) - the exact library used as our "
           "verified baseline. That makes the comparison coherent: same estimator lineage, different pipelines.")
    p(doc, "What the paper proves (its own reported numbers):", bold_=True)
    rows = [
        ["Task in the paper", "Reported result"],
        ["Pairwise causal discovery - Tuebingen benchmark (108 cause-effect pairs)", "97% accuracy (previous best 83%)"],
        ["Edge direction - medical neuropathic-pain graph (475 edges)", "96.2% accuracy (457/475)"],
        ["Counterfactual reasoning - CRASS benchmark", "92.4% (humans 98.2%)"],
        ["Event causality - necessary & sufficient causes (15 vignettes)", ">86% accuracy"],
        ["Normality assessment of actions", "70% - a documented failure mode"],
    ]
    table(doc, rows, widths=[4.2, 2.7])
    p(doc, "What the paper does NOT do (and why that matters for us):", bold_=True)
    bullet(doc, "It does not estimate treatment effects from data - no ATE, no confidence intervals, no "
                "covariate balance, no effect estimation on relational databases.")
    bullet(doc, "It states the limitation explicitly: \"LLMs ignore the actual data\", and it calls for future "
                "work that \"combines LLMs with existing causal techniques\" - which is exactly the architecture "
                "of this project.")
    bullet(doc, "It reports \"unpredictable failure modes\" of LLMs on causal tasks - the same failure class our "
                "guardrails and deterministic estimator are designed to prevent.")
    p(doc, "Therefore the correct way to compare (say this exactly): \"We do not claim to beat their text-"
           "reasoning benchmarks - that is a different task. We claim to complete the pipeline their paper "
           "explicitly leaves open: turning LLM causal understanding into data-grounded effect estimates with "
           "uncertainty and diagnostics. On that layer, we measured our engine against the standard DoWhy "
           "pipeline - the toolchain from the same research group - and we match or beat it on every "
           "apples-to-apples metric.\"")

    doc.add_page_break()

    # ================= 7. HEAD-TO-HEAD =================
    h(doc, "7. The Head-to-Head Evidence (all values measured, none quoted)", 1)
    p(doc, "Everything below was measured locally on the same Olist data with the scripts in benchmarks/. "
           "The baseline is a plain median-split + standard DoWhy propensity_score_weighting pipeline with "
           "default settings - the standard LLM+DoWhy stack.", size=10)
    rows = [
        ["Metric", "Standard baseline", "Our engine", "Result"],
        ["PEHE on known ground truth (lower better)", f"{pehe['naive_diff']['pehe']:.4f} (naive)", f"{OURS_PEHE:.4f}", "90.8% lower error"],
        ["PEHE vs unadjusted OLS", f"{pehe['unadjusted_ols']['pehe']:.4f}", f"{OURS_PEHE:.4f}", "37.9% lower error"],
        ["Valid end-to-end answers (of 7 questions)", "6/7 (photos split degenerates)", "7/7", "full coverage"],
        ["Questions within SMD <= 0.10 balance target", "4/7", f"{n_balanced}/7", "1.5x more balanced"],
        ["Installments question (2-row control trap)", "SMD 1.816, spurious +59.79", "SMD 0.095, +81.93", "19x better balance"],
        ["Mean weighted SMD across questions", f"{base_mean_wsmd:.3f}", f"{ours_mean_wsmd:.3f}", "80% lower imbalance"],
        ["Mean confounding bias removed (per question)", "not reported", f"{mean_bias:.1f}% (up to 99.4%)", "measured on real data"],
        ["Uncertainty quantification", "none", "95% CI + p-value on every estimate", "new capability"],
        ["Significant effects (alpha = 0.05)", "not reported", f"{len(sig)}/7", "verified inference"],
        ["Estimator verification vs DoWhy", "reference", "exact ATE + SMD match on 6/6 shared cases", "gold-standard parity"],
        ["Autonomous causal discovery", "none", "ranked driver-impact alerts", "new capability"],
    ]
    t = table(doc, rows, widths=[2.15, 1.7, 1.75, 1.5])
    for i in range(1, len(rows)):
        shade(t.cell(i, 3), "E2EFDA")

    doc.add_picture(str(FIG_DIR / "fig1_pehe.png"), width=Inches(5.4))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p(doc, "Figure 1 - PEHE: error against known ground truth (lower is better).", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_picture(str(FIG_DIR / "fig2_smd.png"), width=Inches(6.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p(doc, "Figure 2 - SMD before/after weighting on all 7 questions (green line = 0.10 target).", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_picture(str(FIG_DIR / "fig3_degenerate.png"), width=Inches(5.8))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p(doc, "Figure 3 - The degenerate-data kicker: standard split vs our guarded engine.", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)

    h(doc, "7.1 All seven live results (for the appendix of your slides)", 2)
    rows = [["#", "Question", "Rows", "ATE", "95% CI", "p", "SMD raw -> weighted", "Refut."]]
    for i, k in enumerate(ORDER, 1):
        o = olist[k]
        rows.append([str(i), NAMES[k], f"{o['rows']:,}", f"{o['our_ate']:+.4f}",
                     f"[{o['ci_95'][0]:+.3f}, {o['ci_95'][1]:+.3f}]", f"{o['p_value']:.4f}",
                     f"{o['covariate_balance']['raw_smd']:.3f} -> {o['covariate_balance']['weighted_smd']:.3f}",
                     o["refutation_pass_rate"]])
    table(doc, rows, widths=[0.22, 1.7, 0.65, 0.7, 1.0, 0.5, 1.15, 0.5])

    h(doc, "7.2 Engine hardening (show this if asked about methodology fixes)", 2)
    p(doc, "Before the hardening the engine answered 5/6 questions with 1/6 balanced; after, 7/7 answered with "
           "6/7 balanced. The fixes: guarded binarization, DoWhy-aligned propensity clipping [0.05, 0.95], and a "
           "seeded B=50 bootstrap. Full before/after table is in RESULTS.docx (Section 4). The important line: "
           "on the length->freight question, the old extreme weights produced an artifact (-2.46); after clipping "
           "the estimate is +0.09, exactly matching DoWhy.", size=10)

    # ================= 8. WIN / DON'T CLAIM =================
    h(doc, "8. What We Win On - and What We Must Never Claim", 1)
    p(doc, "Safe, strong claims (each backed by a measured number in Section 7):", bold_=True, color=GREEN)
    for i, c in enumerate([
        "90.8% lower estimation error than the naive baseline on known ground truth (PEHE 0.0149 vs 0.1614).",
        "7/7 valid causal answers on 100k+ live rows, vs 6/7 for the standard median-split pipeline.",
        f"{n_balanced}/7 questions reach the gold-standard SMD <= 0.10 balance target; mean confounding bias "
        f"removed {mean_bias:.1f}%, up to 99.4%.",
        "19x better balance on the installments question, where the standard pipeline silently reports a "
        "spurious +59.79 from a 2-row control group.",
        "95% confidence intervals and p-values on every estimate - the comparator paper reports none for "
        "effect estimation.",
        "Exact agreement with the gold-standard DoWhy estimator on all 6 shared cases (verified implementation).",
        "Refutation stress tests and autonomous causal-discovery alerts run automatically on every question.",
        "The pipeline closes the gap the Microsoft paper explicitly identifies: its own conclusion is that LLMs "
        "ignore the data and should be combined with causal techniques - this engine does exactly that, "
        "deterministically.",
    ], 1):
        numbered(doc, c)
    p(doc, "Never claim these (they are false and a sharp examiner will catch them):", bold_=True, color=RED)
    for c in [
        "\"We beat Microsoft Research on their benchmarks.\" - Different task (text reasoning vs effect "
        "estimation). We beat the standard estimation pipeline and complete their open direction.",
        "\"Our effects are proven causal.\" - Observational data: the result holds under the no-unobserved-"
        "confounding assumption; refuters are stress tests, not proof.",
        "\"SMD below 0.10 proves the estimate is correct.\" - It proves the comparison is balanced on measured "
        "confounders; unmeasured ones could remain.",
        "\"Every refutation test passes everywhere.\" - Not true (near-zero effects make strict placebo rules "
        "fail); report per-question results.",
        "\"The LLM derives the statistics.\" - It never computes numbers; keep the neuro-symbolic split explicit.",
    ]:
        bullet(doc, c)

    doc.add_page_break()

    # ================= 9. THE SCRIPT =================
    h(doc, "9. Tomorrow: What to Say and Show (7-Minute Script)", 1)
    h(doc, "9.1 Pre-flight checklist (do this before walking in)", 2)
    for c in [
        "Open terminal in Causal-LLM-Engine; run: .venv/bin/streamlit run app.py --server.headless true "
        "--server.port 8501",
        "Confirm http://localhost:8501 returns the dashboard (HTTP 200).",
        "Confirm .env contains GEMINI_API_KEY (offline cache works without it, but say it is configured).",
        "Have RESULTS.docx and TEAM_BRIEFING.docx open in a second window; have benchmarks/results/ folder "
        "visible as evidence.",
        "Know your fallback line if the model is overloaded: \"The router has a deterministic offline cache "
        "for the canonical questions by design, so the demo never depends on network luck.\"",
    ]:
        numbered(doc, c)
    h(doc, "9.2 The narrative, minute by minute", 2)
    steps = [
        ("0:00-1:00 - Problem", "\"Businesses ask causal questions - 'does more photos raise ratings?' - but ML "
         "learns correlations, and real experiments are impossible on historical data. Existing causal tools "
         "need manual SQL and a statistics PhD; we automated the whole path.\""),
        ("1:00-2:15 - Architecture", "\"Plain English goes into a RAG router: schemas are embedded, the relevant "
         "tables are retrieved, and Gemini returns strict JSON - SQL plus treatment, outcome and confounders. "
         "Here is the key design decision: the LLM never computes a single number. All statistics are "
         "deterministic code, so every result is auditable and reproducible.\""),
        ("2:15-4:00 - Live demo", "Type the photos question. Point at, in order: generated SQL; 110,774 rows; "
         "the SMD card - 'this is our headline trust metric' - 0.076 before weighting, 0.004 after, 94.7% of "
         "bias removed, far inside the 0.10 target; ATE +0.075 stars with 95% CI [0.030, 0.119]; p = 0.0011; "
         "refutation 2/2. Say: 'balanced groups, uncertainty quantified, and the result survived our own "
         "attempts to break it.'"),
        ("4:00-5:00 - Accuracy proof", "Show Figure 1: \"On a synthetic dataset where the true effect is exactly "
         "2.50, our estimator recovers 2.4851 - PEHE 0.0149, a 90.8% error reduction against the naive method. "
         "This proves the math, not just the code.\""),
        ("5:00-6:00 - The kicker", "Show Figure 3 / the installment row: \"Same pipeline, same data, 103,886 "
         "rows. The median is 1 installment, so a standard median split leaves 2 orders in the control group. "
         "The standard pipeline silently reports +59.79 with SMD 1.816. Our guarded split detects this, uses "
         "'1 vs 2+' instead, and reports +81.93 with SMD 0.095 - 19x better balance. We refuse to produce "
         "statistics that the data cannot support.\""),
        ("6:00-7:00 - Honest close", "\"Against the Microsoft Research paper: different task, and we say so - "
         "they measure LLM reasoning on text; they do no effect estimation and explicitly call for combining "
         "LLMs with causal techniques, which is our architecture. Against the standard LLM+DoWhy pipeline on "
         "the same data, we measured: 7/7 valid answers, 6/7 balanced, exact DoWhy agreement, 90.8% lower "
         "PEHE. Full numbers and raw JSON are in RESULTS.docx and benchmarks/results/. Limitations: "
         "observational assumptions, one question retains imbalance, refuters are stress tests not proof.\""),
    ]
    for title, text in steps:
        p(doc, title, bold_=True, size=10.5)
        p(doc, text, size=10)

    # ================= 10. Q&A =================
    h(doc, "10. Q&A Rehearsal - Faculty Questions and Suggested Answers", 1)
    qa = [
        ("What is SMD and why should we trust 0.10?",
         "Standardized Mean Difference measures how different the groups are on each confounder, in standard "
         "deviation units. We compute it before and after weighting. 0.10 is the standard threshold in the "
         "causal-inference literature (Austin 2011; Stuart 2010). After our weighting 6 of 7 questions are "
         "below it - e.g. 0.004 on the photos question - which means the comparison is demonstrably fair on "
         "the measured confounders."),
        ("How do you know your effect is causal and not just correlation?",
         "We do not claim proof from observational data. We estimate the ATE under the explicit assumption of "
         "no unobserved confounding, and we support it with three things: propensity weighting removes measured "
         "confounding (quantified by SMD), the effect is stable under a random added confounder, and a fake "
         "(placebo) treatment collapses to ~0. That is the strongest standard for observational evidence, and "
         "we state the assumption openly."),
        ("Why is PEHE against a synthetic dataset convincing?",
         "Because it is the only setting where the true answer is known. We program an effect of exactly 2.50, "
         "then measure how close each method gets. Ours: 2.4851, within 0.0149. Naive comparison: off by "
         "0.1614. It isolates the estimator's accuracy from every other part of the pipeline."),
        ("Did you beat the Microsoft Research paper?",
         "On their task - LLM causal reasoning on text benchmarks - we do not compete; that is a different "
         "problem. But their paper performs no effect estimation and states that LLMs ignore the data and "
         "should be combined with causal techniques. We built that combination. Against the standard "
         "LLM+DoWhy pipeline (the same toolchain their group created), we measured: 7/7 valid answers vs 6/7, "
         "6/7 balanced vs 4/7, 19x better balance on the degenerate case, 90.8% lower PEHE, and exact "
         "agreement with DoWhy on every shared case."),
        ("Why didn't you run your engine on the paper's datasets?",
         "Their datasets (Tuebingen pairs, vignettes) measure causal reasoning and discovery - they have no "
         "ground-truth treatment effects to estimate, so an ATE engine cannot be scored on them. The common "
         "ground is the estimator layer - DoWhy - where we verified exact agreement. Our evaluation uses a "
         "real relational e-commerce dataset with 100k+ rows and a synthetic ground-truth DGP, which is the "
         "standard way to score an ATE engine."),
        ("Is the LLM actually doing the analysis?",
         "No - deliberately. The LLM only reads English and writes SQL/JSON (plus schema retrieval via "
         "embeddings). Every number - propensity scores, weights, ATE, SMD, CIs - is computed by deterministic "
         "Python and can be re-derived by hand or re-run for an identical result. That is our answer to LLM "
         "hallucination: the model cannot fabricate a statistic because it never touches one."),
        ("What if the LLM writes wrong SQL?",
         "Three defenses: the database layer only permits read-only SELECT/WITH statements; queries run on the "
         "real schema retrieved by RAG; and a deterministic offline cache answers all canonical questions. If "
         "the query returns no data or the treatment cannot support two groups, the engine refuses to "
         "estimate and explains why - it never silently guesses."),
        ("Why binarize continuous variables at all?",
         "Our ATE estimator contrasts two groups. Median-splitting is standard practice for this estimator, "
         "and we made it safe with guards: a strict split when the median sits on a mass point, minimum group "
         "sizes of 30 rows and 1% share, and an explicit refusal otherwise. The installments case shows why "
         "the guard matters - without it you get a 2-row control group."),
        ("Why did some refutation tests fail?",
         "Because they are strict by design. The placebo test expects a random fake treatment to show almost "
         "no effect; when the real effect is near zero, the rule can fail on noise. We report pass rates per "
         "question instead of hiding them - the flagship question passes 100%, and we treat refuters as stress "
         "tests, not proof."),
        ("What are the limitations?",
         "Observational data: no unobserved-confounding guarantee; one question (weight -> freight) retains SMD "
         "0.261; the router can silently map a question to different variables than intended (documented in the "
         "README); and the median split discards dose-response information. We state all of this in the "
         "results document."),
        ("Can we reproduce every number?",
         "Yes, offline, with three commands: run_olist_cases.py, run_baseline_cases.py and "
         "build_results_docx.py. The bootstrap and refuters are seeded (seed 42), and the raw JSON for every "
         "claim is committed in benchmarks/results/."),
        ("What would you do next?",
         "Add a variable-mapping confirmation step to the router, support continuous-treatment estimators "
         "(dose-response instead of binary), add trimming/overlap diagnostics for positivity, and validate on "
         "a second relational dataset."),
    ]
    for i, (q, a) in enumerate(qa, 1):
        p(doc, f"Q{i}. {q}", bold_=True, size=10.5)
        p(doc, f"Suggested answer: {a}", size=10, italic=False)

    doc.add_page_break()

    # ================= 11. GLOSSARY =================
    h(doc, "11. Glossary (for quick lookups)", 1)
    glossary = [
        ("ATE", "Average Treatment Effect: the average causal effect of the treatment on the outcome in the population."),
        ("Treatment / Outcome", "The cause (X) and the effect (Y) named in the question."),
        ("Confounder / Control", "A variable that affects both treatment and outcome; we adjust for it."),
        ("Propensity score e(X)", "P(treatment=1 | confounders), predicted by logistic regression."),
        ("IPTW", "Inverse Probability of Treatment Weighting: re-weights units so groups become comparable."),
        ("Hajek estimator", "Self-normalized IPTW; weights sum to 1; more stable than plain Horvitz-Thompson."),
        ("SMD", "Standardized Mean Difference: group imbalance in SD units; <= 0.10 means balanced."),
        ("Bias reduction %", "(raw SMD - weighted SMD) / raw SMD x 100."),
        ("Positivity / overlap", "Both groups must have a chance of occurring at every confounder value; AUC and clipping guard this."),
        ("Bootstrap", "Resampling to estimate uncertainty; B = 50 resamples, seeded (42)."),
        ("95% CI", "Range containing the true effect in 95% of repeated samples (normal approximation)."),
        ("p-value", "Probability of seeing an effect this large if the true effect were zero."),
        ("Refuters", "DoWhy stress tests: placebo treatment (effect should vanish) and random common cause (effect should be stable)."),
        ("PEHE", "Precision in Estimation of (Heterogeneous) Effect: |estimated - true| when the truth is known."),
        ("RAG", "Retrieval-Augmented Generation: embed schemas, retrieve the relevant ones, then prompt the LLM."),
        ("Neuro-symbolic split", "LLM does language; deterministic code does math. The LLM never computes numbers."),
    ]
    rows = [["Term", "Meaning"]] + [list(g) for g in glossary]
    table(doc, rows, widths=[1.6, 5.3])

    # ================= 12. FILES =================
    h(doc, "12. Files, Commands and Evidence Map", 1)
    rows = [
        ["Path", "What it is"],
        ["app.py", "Streamlit dashboard (Tier 4)."],
        ["ai_engine/schema_embedder.py", "Schema embeddings + cosine RAG retrieval."],
        ["ai_engine/gemini_router.py", "LLM -> strict JSON (sql, treatment, outcome, confounders) + offline cache."],
        ["database/db_setup.py", "SQLite build + read-only query guard."],
        ["math_engine/causal_model.py", "The causal engine: binarization, propensity, Hajek ATE, SMD, bootstrap, refuters, discovery."],
        ["benchmarks/run_olist_cases.py", "Reruns the 7 canonical questions -> olist_cases.json."],
        ["benchmarks/run_baseline_cases.py", "Measures naive / OLS / standard DoWhy baselines -> baseline_cases.json."],
        ["benchmarks/build_results_docx.py", "Rebuilds RESULTS.docx (comparison document)."],
        ["benchmarks/build_team_briefing.py", "Rebuilds this briefing."],
        ["benchmarks/results/", "Raw JSON evidence for every number + paper text extraction + figures."],
        ["RESULTS.docx", "The formal comparison report (tables, figures, before/after hardening)."],
        ["README.md", "Public project documentation, now aligned with the hardened engine."],
    ]
    table(doc, rows, widths=[2.9, 4.0])
    p(doc, "Reproduce everything (offline, no API needed):", bold_=True)
    for cmd in [
        ".venv/bin/python benchmarks/run_olist_cases.py",
        ".venv/bin/python benchmarks/run_baseline_cases.py",
        ".venv/bin/python benchmarks/build_results_docx.py",
        ".venv/bin/python benchmarks/build_team_briefing.py",
    ]:
        code(doc, cmd)
    p(doc, "Run the live app:", bold_=True)
    code(doc, ".venv/bin/streamlit run app.py")

    h(doc, "Integrity Note (read before you present)", 2)
    p(doc, "Every engine number in this pack is seeded (seed 42), reproducible from benchmarks/results/*.json, "
           "and was measured on the same data for both our engine and the baseline. Paper figures are quoted "
           "verbatim from the Microsoft Research PDF included with the repository. Do not inflate these claims "
           "in the presentation - the evidence is already strong, and the honesty about scope is part of what "
           "makes it credible.", size=9, italic=True)

    out = REPO / "TEAM_BRIEFING.docx"
    doc.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
