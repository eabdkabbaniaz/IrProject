
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)
from services.Proccessing.data_loader import DataLoaderService
from sentence_transformers import SentenceTransformer, util
import torch
import os
import joblib
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
from BM25_evaluation import searching 
import torch, joblib, numpy as np, shutil

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
# from services.Proccessing.TextProcessing import TextProcessor

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import joblib
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import random
import os
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import torch, joblib, numpy as np, shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import torch.nn.functional as F
# import os
# import pandas as pd
import csv
from fastapi import HTTPException
import logging

# ==== تحميل الاستعلامات و Qrels ====
def load_queries_txt(path):
    queries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                queries.append({"query_id": parts[0], "query": parts[1]})
    return queries

def load_qrels_json(path):
    qrels = defaultdict(dict)
    with open(path, 'r', encoding='utf-8') as f:
        for item in json.load(f):
            qrels[str(item["query_id"])][str(item["doc_id"])] = int(item["relevance"])
    return qrels

def calculate_precision_recall_at_k(retrieved_docs, relevant_docs, k):
    retrieved_k = retrieved_docs[:k]
    relevant_set = set([doc for doc, rel in relevant_docs.items() if rel > 0])
    retrieved_set = set(retrieved_k)
    tp = len(retrieved_set & relevant_set)
    precision = tp / k if k > 0 else 0
    recall = tp / len(relevant_set) if len(relevant_set) > 0 else 0
    return precision, recall

def calculate_average_precision(retrieved_docs, relevant_docs):
    relevant_set = set([doc for doc, rel in relevant_docs.items() if rel > 0])
    if not relevant_set:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in relevant_set:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / len(relevant_set)

def calculate_mrr(retrieved_docs, relevant_docs):
    relevant_set = set([doc for doc, rel in relevant_docs.items() if rel > 0])
    for i, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in relevant_set:
            return 1 / i
    return 0.0

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)
from services.Proccessing.data_loader import DataLoaderService
from sentence_transformers import SentenceTransformer, util
import torch
import os
import joblib
# from services.BM25.BM25Service import search as bm25_search
from services.Bert.BertService import search as bert_search     
# from services.BM25.BM25Service import search 


def reciprocal_rank_fusion(ranks_list, k=100):
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
        bm25_model,  
        query: str,
        model_dir: str,
        vector_path: str,
        top_k_bm25: int = 25,
        top_k_final: int = 10,
        k_rrf: int = 100
        
):
    """
    - يأخذ أفضل top_k_bm25 وثيقة من BM25
    - يحسب ترتيب BERT‑Embeddings لهذه الوثائق
    - يدمج الترتيبين باستخدام Reciprocal Rank Fusion (RRF)
    - يُرجِع أفضل top_k_final وثيقة
    """

    # ---------- 1) استرجاع BM25 ------------------------------------------
    bm25_hits = searching(query, bm25_model,top_k=top_k_bm25)
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
    fused = reciprocal_rank_fusion([bm25_ranks, bert_ranks], k=10)
    best   = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k_final]
    
    results= [{"doc_id": pid, "rrf_score": round(score, 4)} for pid, score in best]
    return [r["doc_id"] for r in results]




# ==== التقييم الكامل ====
def evaluate_hybird_parallel_represent(bm25_model,queries_path, qrels_path, bert_model,embeddings,top_k=10):
    queries = load_queries_txt(queries_path)
    qrels = load_qrels_json(qrels_path)

    precisions, recalls, maps, mrrs = [], [], [], []

    for query in queries:
        query_id = query['query_id']
        query_text = query['query']
        retrieved_docs =  hybird_parallel_represent(
            bm25_model,
            query_text,
            bert_model,
            embeddings,   
            ) 
        if not retrieved_docs:
            # print(f"❌ Query {query_id} لم يسترجع أي نتائج.")
            continue

        relevant_docs = qrels.get(query_id, {})
        if not relevant_docs:
            continue

        p, r = calculate_precision_recall_at_k(retrieved_docs, relevant_docs, k=top_k)
        ap = calculate_average_precision(retrieved_docs, relevant_docs)
        mrr =calculate_mrr(retrieved_docs, relevant_docs)

        precisions.append(p)
        recalls.append(r)
        maps.append(ap)
        mrrs.append(mrr)

        # print(f"🔍 Query {query_id}: P@{top_k}={p:.2f}, R@{top_k}={r:.2f}, AP={ap:.2f}, MRR={mrr:.2f}")
        # print("✅ Processed query:", retrieved_docs)
        # print("✅ Query terms:", relevant_docs)
    
    return {
        "Model": "hybird_parallel",
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "MAP": np.mean(maps),
        "MRR": np.mean(mrrs)
    }
    # print("\n📊 === المتوسطات ===")
    # print(f"✅ Mean Precision@{top_k}: {np.mean(precisions):.4f}")
    # print(f"✅ Mean Recall@{top_k}: {np.mean(recalls):.4f}")
    # print(f"✅ MAP: {np.mean(maps):.4f}")
    # print(f"✅ MRR: {np.mean(mrrs):.4f}")


# # ==== تشغيل رئيسي ====
# if __name__ == "__main__":
#    evaluate_hybird_parallel_represent(
#     bm25_model=r"C:\Users\user\Downloads\IrProject\IrProject\bm25_model.joblib",
#     queries_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\query100.txt",
#     qrels_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\qrels_quore.json",
#     bert_model=r"C:\Users\user\Downloads\IrProject\IrProject\bert_model",
#     embeddings=r"C:\Users\user\Downloads\IrProject\IrProject\embeddings.joblib",        top_k=10
# )