from services.Proccessing.data_loader import DataLoaderService
from sentence_transformers import SentenceTransformer, util
import torch
import os
import joblib
from services.BM25.BM25Service import search 

def reciprocal_rank_fusion(ranks_list, k=60):
    """
    rrf = sum(1 / (k + rank))
    ranks_list: list of dict {pid: rank}, ranks start at 1
    """
    rrf_scores = {}
    for ranks in ranks_list:
        for pid, rank in ranks.items():
            rrf_scores[pid] = rrf_scores.get(pid, 0) + 1 / (k + rank)
    return rrf_scores


def hybird_parallel_represent(query, documents, model_path, k1=1.5, b=0.75, top_k=10):
    if not os.path.exists(model_path):
        raise ValueError("مسار النموذج غير موجود أو غير صالح.")

    # 1. BM25 Results
    bm25_results = search(query, k1, b, top_k=top_k)
    if not bm25_results:
        return []

    bm25_ranks = {item["doc_id"]: rank + 1 for rank, item in enumerate(bm25_results)}
    top_doc_ids = list(bm25_ranks.keys())

    # 2. Load corresponding documents
    df = DataLoaderService.load(documents)
    df = df[df["ID"].isin(top_doc_ids)]
    df = df.set_index("ID").loc[top_doc_ids]

    texts = df["Processed_Text"].tolist()

    # 3. BERT Similarity
    model = SentenceTransformer(model_path)
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(texts, convert_to_tensor=True)

    similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
    bert_results = [
        {"doc_id": doc_id, "similarity": score.item()}
        for doc_id, score in zip(top_doc_ids, similarities)
    ]
    bert_results = sorted(bert_results, key=lambda x: x["similarity"], reverse=True)
    bert_ranks = {item["doc_id"]: rank + 1 for rank, item in enumerate(bert_results)}

    # 4. Apply Reciprocal Rank Fusion
    fused_scores = reciprocal_rank_fusion([bm25_ranks, bert_ranks])
    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        {"doc_id": doc_id, "rrf_score": round(score, 4)}
        for doc_id, score in sorted_results
    ]