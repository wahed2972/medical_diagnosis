import os
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq

load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_diagnosis(symptoms: list[str]) -> str:
    prompt = f"Patient has symptoms {' ,'.join(symptoms)}. Suggest possible medical diagnosis. Suggest me a proper treatment to cure."
    
    response = client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages = [
            {"role":"system", "content":"You are a helpful medical assistant"},
            {"role":"user","content":prompt}
        ]
    )
    
    return response.choices[0].message.content.strip()


def get_diagnosis_with_context(symptoms, articles):
    context = "\n\n".join([
    f"Title: {a.get('title', '')}\nAbstract: {a.get('abstract', '')}"
    for a in articles[:3] if isinstance(a, dict)
    ])
    print(f"hello ----->{context}")
    print(f"Article --->{articles}")
    prompt = f"""
You are a medical assistant.

Symptoms: {symptoms}

Medical research context:
{context}

Instructions:
- Provide structured output

Format:
1. Most Common Causes
2. Possible Other Causes (from research if relevant)
3. Basic Care Advice
4. When to See a Doctor

Guidelines:
- Treat all symptoms as ONE combined clinical condition
- Only include causes that commonly explain ALL or MOST symptoms together
- PRIORITIZE common infections or general conditions when fever is present

STRICT EXCLUSION RULES:
- DO NOT include conditions that are rare, age-specific, or population-specific
- DO NOT include hospital-acquired, trauma-related, or immune-compromised conditions
- DO NOT include uncommon pathogens or rare disease names
- DO NOT include diseases that require special context (e.g., surgery, ICU, exposure)
- DO NOT include case-report-type diseases even if they match symptoms

DECISION RULE:
- If a cause is not commonly seen in general day-to-day clinical practice → EXCLUDE it
- If unsure whether a cause is common → EXCLUDE it

RESEARCH USAGE:
- Use research ONLY if it clearly supports common, general causes
- If research includes rare or specific diseases → IGNORE that research completely
- If no suitable research remains, return exactly:
  "No additional causes from research"

- Keep output simple and practical
- Add a disclaimer: "This is not a medical diagnosis"
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant"},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()


# output = get_diagnosis(["fever","pain"])
# print(output)