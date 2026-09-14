import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    from ai_engine.schema_embedder import retrieve_relevant_tables
except ModuleNotFoundError:
    from schema_embedder import retrieve_relevant_tables

load_dotenv()
client = genai.Client()

def parse_business_question(user_question: str) -> dict:
    """Retrieves relevant schema via RAG and extracts Causal Parameters into JSON."""
    
    rag_context = retrieve_relevant_tables(user_question, top_k=3)
    
    system_prompt = f"""
    You are an expert Data Scientist and Causal Inference Assistant.
    You have access to a relational SQLite database. Based on semantic search, here are the relevant tables for the user's query:
    
    {rag_context}
    
    Your job is to read the user's business question and output a strictly formatted JSON object.
    Do NOT wrap the output in markdown blocks (e.g., no ```json). Return raw JSON only with these exact keys and format:
    {{
        "sql_query": "A valid SQLite SELECT query that joins the necessary tables and returns the columns for treatment, outcome, and confounders.",
        "treatment": "The exact column name representing the cause.",
        "outcome": "The exact column name representing the effect.",
        "confounders": ["Array of exact column names acting as control variables."],
        "business_hypothesis": "A 1-sentence summary of what we are testing."
    }}
    """
    
    full_prompt = f"{system_prompt}\n\nUser Question: {user_question}"
    
    max_retries = 3
    response = None
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            break
            
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                print(f"Server busy (Attempt {attempt + 1}/{max_retries}). Retrying in 2.5 seconds...")
                time.sleep(2.5)
            else:
                print(f"API Routing Error: {str(e)}")
                return {}
                
    if not response:
        return {}
        
    try:
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        return json.loads(raw_text.strip())
        
    except json.JSONDecodeError:
        print("Error: Model output was not valid JSON.")
        return {}
    except Exception as e:
        print(f"Unexpected parsing error: {str(e)}")
        return {}

if __name__ == "__main__":
    test_question = "Does shipping delay cause lower review scores, controlling for freight price?"
    print(f"Routing Question through RAG pipeline: '{test_question}'\n")
    
    result = parse_business_question(test_question)
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("Routing failed.")