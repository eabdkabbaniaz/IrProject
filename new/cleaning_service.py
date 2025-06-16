from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import pandas as pd
from io import StringIO
from text_processor import TextProcessor, process_text  # الكلاسات تبعك اللي كتبتها فوق

app = FastAPI()
text_processor = TextProcessor()

class Document(BaseModel):
    id: str
    text: str

@app.post("/clean")
def clean_documents(docs: List[Document]):
    cleaned = []
    for doc in docs:
        cleaned_text = process_text(doc.text, text_processor)
        cleaned.append({"id": doc.id, "cleaned_text": cleaned_text})
    return cleaned
