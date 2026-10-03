import os
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq

load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def summarize_pubmed(text: str) -> str:
    prompt = f"Summarize the following medical abstract:\n\n{text}"
    
    response = client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages = [
            {"role":"system", "content":"You are a medical research summarizer."},
            {"role":"user","content":prompt}
        ]
    )
    # print(f"SUMMARY{response}")
    return response.choices[0].message.content.strip()
