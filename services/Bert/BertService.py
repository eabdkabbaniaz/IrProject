from services.Proccessing.data_loader import DataLoaderService
import pandas as pd
import joblib
from datasets import Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import random
import os
import torch

def build_bert_embeddings(documents, model_path):
    df = DataLoaderService.load(documents)
    
    texts = df['Processed_Text'].tolist()
    ids = df['ID'].tolist()

    if model_path is None or not os.path.exists(model_path):
        raise ValueError("يجب أن تقوم بتدريب النموذج أولاً وتزويد المسار الصحيح للنموذج المدرب.")

    model = SentenceTransformer(model_path)
   
    embeddings = model.encode(texts, show_progress_bar=True)

    data = []
    for doc_id, embedding in zip(ids, embeddings):
        data.append({
            "doc_id": doc_id,
            "embedding": embedding.tolist()  
        })

    joblib.dump({
            "data": data,
            "model_path": model_path  
        }, "bert_embeddings.joblib")

    return {"data": data}


def train_bert_model(documents, output_path, epochs=1, batch_size=16):
    df = DataLoaderService.load(documents)
    texts = df['Processed_Text'].tolist()

    examples = [InputExample(texts=[text, text]) for text in texts]

    model = SentenceTransformer('all-MiniLM-L6-v2')

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)

    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=10,
        show_progress_bar=True
    )

    os.makedirs(output_path, exist_ok=True)
    model.save(output_path)

    return f"تم تدريب النموذج وحفظه في: {output_path}"

def search(query, top_k=5):
    if not os.path.exists("bert_embeddings.joblib"):
        raise FileNotFoundError("الملف 'bert_embeddings.joblib' غير موجود. يرجى إنشاء التمثيلات أولاً.")

    bert_data = joblib.load("bert_embeddings.joblib")
    data = bert_data["data"]
    model_path = bert_data.get("model_path")

    if model_path is None or not os.path.exists(model_path):
        raise ValueError("مسار النموذج غير موجود أو غير معرف داخل بيانات التمثيل.")

    model = SentenceTransformer(model_path)

    query_embedding = model.encode(query, convert_to_tensor=True)

    doc_embeddings = [item["embedding"] for item in data]
    doc_embeddings_tensor = torch.tensor(doc_embeddings)

    similarities = util.cos_sim(query_embedding, doc_embeddings_tensor)[0]

    top_results = torch.topk(similarities, k=top_k)

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        results.append({
            "doc_id": data[idx]["doc_id"],
            "score": round(score.item(), 4)
        })

    return results