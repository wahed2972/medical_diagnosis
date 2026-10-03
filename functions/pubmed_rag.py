import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer("all-MiniLM-L6-v2")

def fetch_pubmed_rag(query: str, max_results=20):
    headers = {"User-Agent": "Mozilla/5.0"}

    generic_query = f"{query} causes diagnosis treatment review"

    # -------------------------
    # STEP 1: FETCH PUBMED DATA
    # -------------------------
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    search_params = {
        "db": "pubmed",
        "term": generic_query,
        "retmax": max_results,
        "retmode": "json"
    }

    search_response = requests.get(search_url, params=search_params, headers=headers).json()
    id_list = search_response["esearchresult"]["idlist"]

    if not id_list:
        return []

    ids = ",".join(id_list)

    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    fetch_params = {
        "db": "pubmed",
        "id": ids,
        "retmode": "xml"
    }

    fetch_response = requests.get(fetch_url, params=fetch_params, headers=headers)
    soup = BeautifulSoup(fetch_response.text, "lxml")

    articles_xml = soup.find_all("pubmedarticle")

    # -------------------------
    # STEP 2: BUILD DOCUMENTS
    # -------------------------
    docs = []
    metadata = []

    for article, pmid in zip(articles_xml, id_list):

        title_tag = article.find("articletitle")
        abstract_tag = article.find("abstract")

        title = title_tag.get_text(strip=True) if title_tag else ""
        abstract = abstract_tag.get_text(" ", strip=True) if abstract_tag else ""

        text = title + " " + abstract

        if len(text.strip()) == 0:
            continue

        docs.append(text)
        metadata.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract
        })

    if not docs:
        return []

    # -------------------------
    # STEP 3: EMBEDDINGS
    # -------------------------
    doc_embeddings = model.encode(docs)

    query_embedding = model.encode([query])

    # -------------------------
    # STEP 4: FAISS SEARCH
    # -------------------------
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(np.array(doc_embeddings))

    scores, indices = index.search(np.array(query_embedding), k=3)

    # -------------------------
    # STEP 5: RETURN TOP K
    # -------------------------
    results = []

    for idx in indices[0]:
        results.append(metadata[idx])

    return results