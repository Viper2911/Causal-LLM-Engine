import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ai_engine.gemini_router import parse_business_question
q = "Does shipping freight cause lower review scores, controlling for price and product weight?"
res = parse_business_question(q)
print(json.dumps(res, indent=2))
