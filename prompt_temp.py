import connect_db
from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate


prompt = f"""
You are a hospital profit optimization system.

Your goal:
1. Select CPT code combinations that the doctor typically performs
2. Generate 3 to 4 different CPT package combinations
3. Maximize total revenue for each package
4. Ensure all CPTs are clinically relevant to the patient's condition

STRICT RULES:
- Use ONLY the provided context
- Do NOT invent CPT codes
- Prefer CPT codes associated with higher revenue cases
- Each package can contain ANY number of CPT codes (not fixed)
- Each package must be a realistic combination actually used together
- Avoid duplicate or highly similar packages
- Sort packages in descending order of total_estimated_revenue

Return ONLY valid JSON:

{{
  "doctor_id": {connect_db.doc_id},
  "packages": [
    {{
      "cpt_codes": ["CPT1", "CPT2", "CPT3"],
      "total_estimated_revenue": number
    }},
    {{
      "cpt_codes": ["CPT4", "CPT5"],
      "total_estimated_revenue": number
    }},
    {{
      "cpt_codes": ["CPT6", "CPT7", "CPT8", "CPT9"],
      "total_estimated_revenue": number
    }}
  ],
  "reason": "packages selected based on high revenue and relevance to symptoms",
  "confidence": "high/medium/low"
}}

Context:
{connect_db.results}
"""

llm= ChatOpenAI(
    model="gpt-5-mini"
)

answer = None

def generate_answer():
    global answer
    answer=llm.invoke(prompt)
    
    return answer.content
# print(answer.content)