from services.Proccessing.data_loader import DataLoaderService
from sentence_transformers import SentenceTransformer, util
import torch
import os
import joblib

from services.BM25.BM25Service import search 

def hybird_sequential_represent(query, documents, model_path, k1, b, top_k=5):

    if not os.path.exists(model_path):
        raise ValueError("مسار النموذج غير موجود أو غير صالح.")

    bm25_results = search(query, k1, b, top_k=top_k)
    if not bm25_results:
        return []

    top_doc_ids = [item["doc_id"] for item in bm25_results]

    df = DataLoaderService.load(documents)  
    df = df[df["ID"].isin(top_doc_ids)]
    df = df.set_index("ID").loc[top_doc_ids]  

    texts = df["Processed_Text"].tolist()

    # 3. BERT similarity
    model = SentenceTransformer(model_path)

    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(texts, convert_to_tensor=True)

    similarities = util.cos_sim(query_embedding, doc_embeddings)[0]

    results = []
    for doc_id, score in zip(top_doc_ids, similarities):
        results.append({
            "doc_id": doc_id,
            "similarity_score": round(score.item(), 4)
        })

    results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)
    return results
