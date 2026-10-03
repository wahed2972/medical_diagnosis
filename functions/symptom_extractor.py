import re
from groq import Groq
from dotenv import load_dotenv
import os
import ast

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_symptoms(text: str) -> list[str]:
    symptoms = re.findall(r"\b(headache|fever|nausea|fatigue|pain)\b",text.lower())
    return list(set(symptoms))

import json
import re

def extract_symptoms_llm(text):
    
    prompt = f"""
You are a medical extraction system.

Extract symptoms from the text.

🚨 RULES (STRICT):
- Return ONLY valid JSON array
- Do NOT write explanations
- Do NOT write sentences
- Do NOT use markdown
- Do NOT add bullet points
- Do NOT say anything like "here is"

FORMAT EXAMPLE:
["fever", "headache", "pain"]

TEXT:
{text}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content.strip()

    # 🔥 Extract only JSON part if extra text appears
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        content = match.group()

    return json.loads(content)


text = "I am having back pain and having fever too along with that having cold and cough. Sometimes headache also with sneeze"

symptoms = extract_symptoms_llm(text)
# print(symptoms)

