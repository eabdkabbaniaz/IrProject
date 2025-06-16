from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

app = FastAPI()

class CleanedDoc(BaseModel):
    id: str
    cleaned_text: str

vectorizer = TfidfVectorizer(max_df=0.7, min_df=0.01)

@app.post("/tfidf")
def build_tfidf_matrix(docs: List[CleanedDoc]):
    texts = [doc.cleaned_text for doc in docs]
    doc_ids = [doc.id for doc in docs]

    matrix = vectorizer.fit_transform(texts)

    # Save matrix and vectorizer
    joblib.dump(matrix, "tfidf_matrix.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    return {
        "message": "TF-IDF matrix created",
        "shape": matrix.shape,
        "doc_ids": doc_ids
    }
