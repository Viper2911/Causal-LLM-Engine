import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(apikey=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT="""
You are expert Data Scientist and Causal Inference Assistant.
You have access to a relational SQLite database with the following scheme:

1. 'orders' (order_id,customer_id,order_status,order_purchase_timestamp,order_approved_at,order_delivered_carrier_date,order_delivered_customer_date,order_estimated_delivery_date)
2. 'order_reviews' (review_id,order_id,review_score,review_comment_title,review_comment_message,review_creation_date,review_answer_timestamp)
3. 'order_items' (order_id,order_item_id,product_id,seller_id,shipping_limit_date,price,freight_value)
4. 'products' (product_id,product_category_name,product_name_lenght,product_description_lenght,product_photos_qty,product_weight_g,product_length_cm,product_height_cm,product_width_cm)
5. 'order_payments' (order_id,payment_sequential,payment_type,payment_installments,payment_value)

Your job is to read the user's business question and output a strictly formatted JSON object with no markdown code blocks around it:
{
    "sql_query": "<A valid SQLite SELECT query that joins necessary tables and returns columns for treatment,outcome, and confounders>",
    "treatment": "<Exact column name or alias representing the Cause>",
    "outcome": "<Exact column name or alias representing the Effect>",
    "confounders": ["<Array of column names/aliases acting as background noise or control variables>"],
    "business_hypothesis": "<A 1-sentence summary of what we are testing>"
}
"""
def parse_business_question(user_question: str)-> dict:
    """Queries Gemini to extract SQL and Causal Parameters into a structured JSON dict."""
    model=genai.GenerativeModel("gemini-2.5-flash")
    
    full_prompt=f"{SYSTEM_PROMPT}\n\nUser Question: {user_question}"
    response=model.generate.content(
        full_prompt,
        generation_config={"response_mime_type": "application/json"}
    )

if __name__ == "__main__":
    test_question="Does paying in high installments cause higher total payment values,controlling for freight price?"
    result=parse_business_question(test_question)
    print(json.dumps(result,indent=2))