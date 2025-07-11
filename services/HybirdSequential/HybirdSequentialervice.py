from services.Proccessing.data_loader import DataLoaderService
from sentence_transformers import SentenceTransformer, util
import torch
import os
import joblib
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
from services.BM25.BM25Service import search 

# def hybird_sequential_represent(query, documents, model_path, top_k=5):

#     if not os.path.exists(model_path):
#         raise ValueError("مسار النموذج غير موجود أو غير صالح.")

#     bm25_results = search(query, top_k=top_k)
#     if not bm25_results:
#         return []

#     top_doc_ids = [item["doc_id"] for item in bm25_results]

#     df = DataLoaderService.load(documents)  
#     df = df[df["ID"].isin(top_doc_ids)]
#     df = df.set_index("ID").loc[top_doc_ids]  

#     texts = df["Processed_Text"].tolist()

#     # 3. BERT similarity
#     model = SentenceTransformer(model_path)

#     query_embedding = model.encode(query, convert_to_tensor=True)
#     doc_embeddings = model.encode(texts, convert_to_tensor=True)

#     similarities = util.cos_sim(query_embedding, doc_embeddings)[0]

#     results = []
#     for doc_id, score in zip(top_doc_ids, similarities):
#         results.append({
#             "doc_id": doc_id,
#             "similarity_score": round(score.item(), 4)
#         })

#     results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)
#     return results

# -----------------------------------------------------------
# hybrid search: BM25  ➜  ثم BERT لإعادة الترتيب
# -----------------------------------------------------------
def hybrid_sequential_represent(
    query: str,
    model_dir: str,                # مجلد النموذج المحفوظ بـ save_pretrained
    vector_path: str,              # نفس vector_path الذي مررته لـ train_bert_model
    top_k_bm25: int = 25,          # عدد النتائج من BM25 قبل إعادة الترتيب
    top_k_final: int = 10           # عدد النتائج النهائية بعد BERT
) -> List[Dict]:

    # -------- 1) مرحلة BM25 -----------------------------------------------
    bm25_hits = search(query, top_k=top_k_bm25)
    if not bm25_hits:
        return []

    top_doc_ids = [hit["doc_id"] for hit in bm25_hits]

    # -------- 2) تحميل التضمينات المحفوظة ----------------------------------
    vec_path = Path(vector_path)
    emb_matrix = joblib.load(vec_path)                       # (N_docs, H)
    doc_ids = joblib.load(vec_path.with_name(vec_path.name + "_doc_ids"))

    # اصنع خريطة معرّف → فهرس ليسهل الاسترجاع
    id2idx = {d: i for i, d in enumerate(doc_ids)}
    missing = [d for d in top_doc_ids if d not in id2idx]
    if missing:
        raise ValueError(f"معرّفات مفقودة في ملف التضمينات: {missing}")

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

    return results
