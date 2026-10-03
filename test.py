# import requests

# try:
#     r = requests.get(
#         "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
#         params={"db": "pubmed", "term": "pain", "retmax": 1, "retmode": "json"},
#         timeout=10
#     )
#     print("Status:", r.status_code)
#     print("Response:", r.text[:200])
# except Exception as e:
#     print("Error:", e)

from fastapi import FastAPI
from pydantic import BaseModel
from functions.symptom_extractor import extract_symptoms
from functions.diagnosis_symptom import get_diagnosis,get_diagnosis_with_context
from functions.pubmed_articles import fetch_pubmed_articles_with_metadata
from functions.summarize_pubmed import summarize_pubmed
from functions.pubmed_rag import fetch_pubmed_rag


# app = FastAPI()

# class SymptomInput(BaseModel):
#     descrption: str

# @app.post("/diagnosis")
def diagnosis(data:str):
    symptom = extract_symptoms(data)
    # diagnosis_result = get_diagnosis(symptom)
    pubmed_article = fetch_pubmed_articles_with_metadata(" ".join(symptom))
    # pubmed_article = fetch_pubmed_rag(" ".join(symptom))
    summary = summarize_pubmed(pubmed_article[:3])
    diagnosis_result = get_diagnosis_with_context(symptom,pubmed_article)
    
    return {
        "symptom":symptom,
        "diagnosis":diagnosis_result
        # "pubmed_summary":summary
    }
    
result = diagnosis("I am having cold and cough")
print(result)