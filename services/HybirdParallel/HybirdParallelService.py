from services.Proccessing.data_loader import DataLoaderService
from sentence_transformers import SentenceTransformer, util
import torch
import os
import joblib
from services.BM25.BM25Service import search as bm25_search
from services.Bert.BertService import search as bert_search     


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

def hybird_parallel_represent(
        query: str,
        model_dir: str,
        vector_path: str,
        top_k_bm25: int = 25,
        top_k_final: int = 10,
        k_rrf: int = 60
):
    """
    - يأخذ أفضل top_k_bm25 وثيقة من BM25
    - يحسب ترتيب BERT‑Embeddings لهذه الوثائق
    - يدمج الترتيبين باستخدام Reciprocal Rank Fusion (RRF)
    - يُرجِع أفضل top_k_final وثيقة
    """

    # ---------- 1) استرجاع BM25 ------------------------------------------
    bm25_hits = bm25_search(query, top_k=top_k_bm25)
    if not bm25_hits:
        return []

    bm25_ranks = {hit["doc_id"]: rank + 1
                  for rank, hit in enumerate(bm25_hits)}
    top_doc_ids = list(bm25_ranks.keys())

    # ---------- 2) استرجاع ترتيب BERT لنفس الوثائق -----------------------
    # نستدعى bert_search مع نفس vector_path/model_dir ثم نُبقى فقط ما يخص top_doc_ids
    bert_hits_all = bert_search(query,
                                vector_path=vector_path,
                                model_dir=model_dir,
                                top_k=len(top_doc_ids))       # نحصل على كل النتائج الممكنة

    bert_ranks = {}
    rank = 1
    for hit in bert_hits_all:
        pid = hit["doc_id"]
        if pid in top_doc_ids:          # نأخذ فقط المشترَك مع قائمة BM25
            bert_ranks[pid] = rank
            rank += 1
    # إذا كان بعض الـ pids لم يأتِ من bert  نعطيه رتبة كبيرة (بعد الأخير)
    max_rank = rank
    for pid in top_doc_ids:
        if pid not in bert_ranks:
            max_rank += 1
            bert_ranks[pid] = max_rank

    # ---------- 3) دمج RRF ----------------------------------------------
    fused = reciprocal_rank_fusion([bm25_ranks, bert_ranks], k=k_rrf)
    best   = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k_final]

    return [{"doc_id": pid, "rrf_score": round(score, 4)} for pid, score in best]
