from fastapi import FastAPI
from pydantic import BaseModel
from functions.symptom_extractor import extract_symptoms,extract_symptoms_llm
from functions.diagnosis_symptom import get_diagnosis,get_diagnosis_with_context
from functions.pubmed_articles import fetch_pubmed_articles_with_metadata
from functions.summarize_pubmed import summarize_pubmed


app = FastAPI()

class SymptomInput(BaseModel):
    descrption: str

@app.post("/diagnosis")
def diagnosis(data:SymptomInput):
    symptom = extract_symptoms_llm(data.descrption)
    # diagnosis_result = get_diagnosis(symptom)
    pubmed_article = fetch_pubmed_articles_with_metadata(" ".join(symptom))
    summary = summarize_pubmed(pubmed_article[:3])
    diagnosis_result = get_diagnosis_with_context(symptom,pubmed_article)
    
    return {
        "symptom":symptom,
        "diagnosis":diagnosis_result,
        "pubmed_summary":summary
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app",host="127.0.0.1",port=8080, reload=True)