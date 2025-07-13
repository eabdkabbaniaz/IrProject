
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

def hybrid_sequential_represent(
    bm25_model,
    query: str,
    model_dir: str,                # مجلد النموذج المحفوظ بـ save_pretrained
    vector_path: str,              # نفس vector_path الذي مررته لـ train_bert_model
    top_k_bm25: int = 25,          # عدد النتائج من BM25 قبل إعادة الترتيب
    top_k_final: int = 10           # عدد النتائج النهائية بعد BERT
) -> List[Dict]:
    print(1)

    # -------- 1) مرحلة BM25 -----------------------------------------------
    bm25_hits = searching(query,bm25_model, top_k=top_k_bm25)
    if not bm25_hits:
        return []
    print(1)
    top_doc_ids = [str(hit["doc_id"]) for hit in bm25_hits]

    # -------- 2) تحميل التضمينات المحفوظة ----------------------------------
    vec_path = Path(vector_path)
    emb_matrix = joblib.load(vec_path)                       # (N_docs, H)
    doc_ids = joblib.load(vec_path.with_name(vec_path.name + "_doc_ids"))
    doc_ids = [str(d) for d in doc_ids]
    print(1)

    # اصنع خريطة معرّف → فهرس ليسهل الاسترجاع
    id2idx = {d: i for i, d in enumerate(doc_ids)}
    missing = [d for d in top_doc_ids if d not in id2idx]
    if missing:
        raise ValueError(f"معرّفات مفقودة في ملف التضمينات: {missing}")
    print(1)

    # التضمينات المقابلة لوثائق BM25 (ترتيب مطابق لـ top_doc_ids)
    doc_embs = torch.tensor(
        np.stack([emb_matrix[id2idx[pid]] for pid in top_doc_ids]),
        dtype=torch.float32
    )

    # -------- 3) حساب تمثيل الاستعلام بنفس الأسلوب ------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModel.from_pretrained(model_dir)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        last_hidden  = model(**inputs).last_hidden_state      # (1, T, H)
        attn_mask    = inputs["attention_mask"]               # (1, T)

        mask   = attn_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)         # (1, H)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        query_emb = summed / counts                           # (1, H)

    # -------- 4) حساب تشابه كوزاين -----------------------------------------
    sims = util.cos_sim(query_emb, doc_embs)[0]               # (top_k_bm25,)

    # -------- 5) إعداد النتائج ---------------------------------------------
    results = [
        {
            "doc_id": pid,
            "similarity_score": round(float(score), 4),
            "bm25_score": next(hit["score"] for hit in bm25_hits if hit["doc_id"] == pid)
        }
        for pid, score in zip(top_doc_ids, sims)
    ]

    # إعادة الترتيب بناءً على تشابه BERT (يمكنك دمج الدرجتين إن أردت)
    results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)[:top_k_final]
    # results = [str(doc_ids[int(idx)]) for idx in top_k_final]

    # print(results)
    return [r["doc_id"] for r in results]




# ==== التقييم الكامل ====
def evaluate_hybrid_sequential_represent(bm25_model,bert_model,embeddings,queries_path, qrels_path, top_k=10):
    queries = load_queries_txt(queries_path)
    qrels = load_qrels_json(qrels_path)

    precisions, recalls, maps, mrrs = [], [], [], []

    for query in queries:
        query_id = query['query_id']
        query_text = query['query']
        retrieved_docs =  hybrid_sequential_represent(
            bm25_model,
            query_text,
            bert_model,
            embeddings,   
            ) 
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

        # print(f"🔍 Query {query_id}: P@{top_k}={p:.2f}, R@{top_k}={r:.2f}, AP={ap:.2f}, MRR={mrr:.2f}")
    #     print("✅ Processed query:", retrieved_docs)
    #     print("✅ Query terms:", relevant_docs)
    # print("\n📊 === المتوسطات ===")
    # print(f"✅ Mean Precision@{top_k}: {np.mean(precisions):.4f}")
    # print(f"✅ Mean Recall@{top_k}: {np.mean(recalls):.4f}")
    # print(f"✅ MAP: {np.mean(maps):.4f}")
    # print(f"✅ MRR: {np.mean(mrrs):.4f}")
    return {
        "Model": "hybird_sequential",
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "MAP": np.mean(maps),
        "MRR": np.mean(mrrs)
    }
# ==== تشغيل رئيسي ====
if __name__ == "__main__":
    # evaluate_hybrid_sequential_represent(
    #     queries_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\queries_quore.txt",
    #     qrels_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\qrels_quore.json",
    #     top_k=10
    # )
    # evaluate_hybrid_sequential_represent(
    #     bm25_model=r"C:\Users\user\Downloads\IrProject\IrProject\bm25_model.joblib",
    #     bert_model=r"C:\Users\user\Downloads\IrProject\IrProject\bert_model",
    #     embeddings=r"C:\Users\user\Downloads\IrProject\IrProject\embeddings.joblib",  
    #     queries_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\query100.txt",
    #     qrels_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\qrels_quore.json",
    #     top_k=10
    # )
    
    # evaluate_hybrid_sequential_represent(
    #     bm25_model=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\files\bm25_model.joblib",
    #     bert_model=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\files\bert_model",
    #     embeddings=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\files\embeddings.joblib",
    #     queries_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\antique_train_queries.txt",
    #     qrels_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\antique_train_qrels.json",
  
  
    # )
    evaluate_hybrid_sequential_represent(
        bm25_model=r"C:\Users\user\Downloads\IrProject\IrProject\bm25_model.joblib",
        bert_model=r"C:\Users\user\Downloads\IrProject\IrProject\bert_model",
        embeddings=r"C:\Users\user\Downloads\IrProject\IrProject\embeddings.joblib",  
        queries_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\query100.txt",
        qrels_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\qrels_quore.json",
        top_k=10
 )