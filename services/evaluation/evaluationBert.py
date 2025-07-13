import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

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
def search(
    query: str,
    vector_path: str ,   
    model_dir: str,                 
    top_k: int = 5
) -> List[Dict]:

    # ------------------------------------------------------------------ #
    # 1) تحميل التضمينات ومعرّفات الوثائق
    # ------------------------------------------------------------------ #
    vec_path = Path(vector_path)
    if not vec_path.is_file():
        raise FileNotFoundError(
            f"ملف التضمينات '{vec_path}' غير موجود؛ درّب النموذج أو تحقق من المسار."
        )

    emb_matrix = joblib.load(vec_path)                         # np.ndarray (N, H)

    doc_ids_path = vec_path.with_name(vec_path.name + "_doc_ids")
    if not doc_ids_path.is_file():
        raise FileNotFoundError(
            f"ملف معرّفات الوثائق '{doc_ids_path}' غير موجود."
        )

    doc_ids = joblib.load(doc_ids_path)                        # List[str|int]

    # ------------------------------------------------------------------ #
    # 2) تحميل النموذج والـ Tokenizer المُخزَّنين بـ save_pretrained
    # ------------------------------------------------------------------ #
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"مجلد النموذج '{model_dir}' غير موجود."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModel.from_pretrained(model_dir)
    model.eval()

    # ------------------------------------------------------------------ #
    # 3) حساب تمثيل الاستعلام (Mean‑Pooling + تطبيع ℓ2)
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        inputs = tokenizer(
            query, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        )

        outputs        = model(**inputs)
        last_hidden    = outputs.last_hidden_state           # (1, T, H)
        attention_mask = inputs["attention_mask"]            # (1, T)

        mask   = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)        # (1, H)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        query_emb = summed / counts                          # (1, H)
        query_emb = F.normalize(query_emb, p=2, dim=1)       # ℓ2‑norm

    # ------------------------------------------------------------------ #
    # 4) حساب التشابه الكوني مع جميع التضمينات المُخزَّنة
    # ------------------------------------------------------------------ #
    doc_embs = torch.tensor(emb_matrix, dtype=query_emb.dtype)
    doc_embs = F.normalize(doc_embs, p=2, dim=1)              # (N, H)

    sims = torch.matmul(query_emb, doc_embs.T).squeeze(0)     # (N,)

    top_k = min(top_k, sims.size(0))
    top_scores, top_indices = torch.topk(sims, k=top_k)

    # ------------------------------------------------------------------ #
    # 5) إعداد النتائج النهائية
    # ------------------------------------------------------------------ #
    # results = [
    #     {
    #         "doc_id": doc_ids[int(idx)],
    #         "score" : round(score.item(), 4)
    #     }
    #     for score, idx in zip(top_scores, top_indices)
    # ]
    results = [str(doc_ids[int(idx)]) for idx in top_indices]
        
    # print(results)
    return results

# ==== التقييم الكامل ====
def Bert_evaluate_search_engine(queries_path, qrels_path, top_k,bert_model,embeddings):
    queries = load_queries_txt(queries_path)
    qrels = load_qrels_json(qrels_path)

    precisions, recalls, maps, mrrs = [], [], [], []

    for query in queries:
        query_id = query['query_id']
        query_text = query['query']
        retrieved_docs =  search(query_text , embeddings, bert_model)

        if not retrieved_docs:
            print(f"❌ Query {query_id} لم يسترجع أي نتائج.")
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

    #     print(f"🔍 Query {query_id}: P@{top_k}={p:.2f}, R@{top_k}={r:.2f}, AP={ap:.2f}, MRR={mrr:.2f}")
    #     print("✅ Processed query:", retrieved_docs)
    #     print("✅ Query terms:", relevant_docs)
    # print("\n📊 === المتوسطات ===")
    # print(f"✅ Mean Precision@{top_k}: {np.mean(precisions):.4f}")
    # print(f"✅ Mean Recall@{top_k}: {np.mean(recalls):.4f}")
    # print(f"✅ MAP: {np.mean(maps):.4f}")
    # print(f"✅ MRR: {np.mean(mrrs):.4f}")
    return {
        "Model": "Bert",
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "MAP": np.mean(maps),
        "MRR": np.mean(mrrs)
    }
# ==== تشغيل رئيسي ====
# if __name__ == "__main__":
#     Bert_evaluate_search_engine(
#         queries_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\queries_quore.txt",
#         qrels_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\qrels_quore.json",
#         top_k=10,
#         bert_model=r"C:\Users\user\Downloads\IrProject\IrProject\bert_model",
#         embeddings=r"C:\Users\user\Downloads\IrProject\IrProject\embeddings.joblib",
#     )