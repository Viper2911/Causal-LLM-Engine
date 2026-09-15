import streamlit as st
import pandas as pd

from database.db_setup import execute_query
from ai_engine.gemini_router import parse_business_question
from math_engine.causal_model import run_causal_analysis

st.set_page_config(page_title="Causal AI Engine", layout="wide")

st.title("Enterprise Causal AI Engine")
st.markdown("Autonomously translates natural language to SQL via RAG and mathematically verifies the Average Treatment Effect (ATE) using our custom IPTW algorithm.")

user_query = st.text_input(
    "Ask a causal business question:", 
    placeholder="e.g., Does paying in high installments cause higher total payment values, controlling for sequential payments?"
)

if st.button("Analyze Causal Impact"):
    if not user_query:
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Executing RAG Vector Search & AI Parameter Extraction..."):
            ai_response = parse_business_question(user_query)
            
        if not ai_response or "sql_query" not in ai_response:
            st.error("AI Routing Failed: Could not extract valid JSON. Check terminal for API errors.")
        else:
            st.success("AI successfully mapped variables via RAG and generated SQL.")
            
            if "business_hypothesis" in ai_response:
                st.info(f"**Testing Hypothesis:** {ai_response['business_hypothesis']}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Treatment (Cause)", ai_response.get("treatment", "N/A"))
            col2.metric("Outcome (Effect)", ai_response.get("outcome", "N/A"))
            col3.write(f"**Confounders (Control):** {', '.join(ai_response.get('confounders', []))}")
            
            with st.expander("View Generated SQL Query"):
                st.code(ai_response["sql_query"], language="sql")
                
            with st.spinner("Executing multi-table JOIN query on Olist SQLite database..."):
                df = execute_query(ai_response["sql_query"])
            
            if df.empty:
                st.error("SQL query returned no data. The AI may have hallucinated a column, or the JOIN produced zero rows.")
            else:
                st.success(f"Fetched {len(df)} rows from the live database.")
                with st.expander("View Raw Data Preview"):
                    st.dataframe(df.head())
                
                with st.spinner("Calculating Custom Algorithmic ATE via Propensity Score Weighting..."):
                    math_result = run_causal_analysis(
                        df=df,
                        treatment=ai_response["treatment"],
                        outcome=ai_response["outcome"],
                        confounders=ai_response["confounders"]
                    )
                
                if math_result["status"] == "error":
                    st.error(f"Math Engine Error: {math_result['message']}")
                else:
                    st.success("Custom Proprietary Math Complete!")
                    
                    st.markdown("### The Causal Impact (ATE)")
                    st.metric(
                        label=f"Custom Proprietary ATE (Impact of {ai_response['treatment']} on {ai_response['outcome']})", 
                        value=math_result["custom_ate"]
                    )
                    
                    if math_result.get("alternative_causes"):
                        st.warning("💡 **Causal Discovery Alert: Stronger Confounders Found!**")
                        for alt in math_result['alternative_causes']:
                            st.write(f"- Variable **'{alt['variable']}'** has a stronger impact score ({alt['confounder_impact']}) than the requested treatment ({alt['treatment_impact']}).")

                    with st.expander("View Statistical Refutation Tests (Panel Defense)"):
                        st.text("PLACEBO REFUTER (Target ATE = 0.0)")
                        st.text(math_result.get("placebo_summary", "Not available"))
                        st.text("---")
                        st.text("RANDOM CAUSE REFUTER")
                        st.text(math_result.get("random_cause_summary", "Not available"))