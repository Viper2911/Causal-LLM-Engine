import os
import json
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
    Do NOT wrap the output in markdown blocks (e.g., no ```json). Return raw JSON only:
    {{
        "sql_query": "<A SELECT SQLite fetch joins necessary query tables that the to variables>",
        "treatment": "<Exact Cause column name representing the>",
        "outcome": "<Exact Effect column name representing the>",
        "confounders": ["<Array acting as background column exact names of variables>"],
        "business_hypothesis": "<A 1-sentence are of summary testing we what>"
    }}
    """
    
    full_prompt = f"{system_prompt}\n\nUser Question: {user_question}"
    
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
        
    except json.JSONDecodeError:
        print("Error: Model output was not valid JSON. Hallucination occurred.")
        return {}
    except Exception as e:
        print(f"API Routing Error: {str(e)}")
        return {}

if __name__ == "__main__":
    test_question = "Does shipping delay cause lower review scores, controlling for freight price?"
    print(f"Routing Question through RAG pipeline: '{test_question}'\n")
    
    result = parse_business_question(test_question)
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("Routing failed.")