from services.Proccessing.data_loader import DataLoaderService
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

def train_bert_model(
    documents: str,
    output_path: str,
    vector_path: str,
    batch_size: int = 8,
    model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
) -> str:

    # --- 1) تحميل البيانات -----------------------------------------------------
    df = DataLoaderService.load(documents)[["text", "doc_id"]]
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)

    texts    = df["text"].tolist()
    doc_ids  = df["doc_id"].tolist()
    if not texts:
        raise ValueError("لا توجد نصوص صالحة بعد التنظيف!")

    # --- 2) تحميل النموذج والـ tokenizer --------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)
    model.eval()

    # --- 3) توليد التضمينات ---------------------------------------------------
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            outputs          = model(**inputs)
            last_hidden      = outputs.last_hidden_state
            attention_mask   = inputs["attention_mask"]

            mask   = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            batch_embeddings = (summed / counts).cpu().numpy()
            embeddings.append(batch_embeddings)

            print(
                f"Processed batch {i // batch_size + 1} / "
                f"{(len(texts) + batch_size - 1) // batch_size}"
            )

    embeddings = np.vstack(embeddings)

    # --- 4) تنظيف مسارات الحفظ السابقة ----------------------------------------
    vector_path   = Path(vector_path)
    output_path   = Path(output_path)

    # احذف ملفات التضمينات القديمة إن وُجدت
    for f in [
        vector_path,
        vector_path.with_name(vector_path.name + "_doc_ids")
    ]:
        if f.exists():
            f.unlink()

    # احذف مجلد النموذج القديم بالكامل
    if output_path.exists():
        shutil.rmtree(output_path)

    # --- 5) إنشاء المجلدات من جديد -------------------------------------------
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- 6) الحفظ -------------------------------------------------------------
    joblib.dump(embeddings, vector_path)
    joblib.dump(doc_ids, vector_path.with_name(vector_path.name + "_doc_ids"))

    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"✅ Embeddings saved to   : {vector_path}")
    print(f"✅ Doc‑IDs   saved to   : {vector_path.with_name(vector_path.name + '_doc_ids')}")
    print(f"✅ Model & tokenizer to : {output_path}")

    return str(output_path)


# def search(query, top_k=5):
#     if not os.path.exists("bert_embeddings.joblib"):
#         raise FileNotFoundError("الملف 'bert_embeddings.joblib' غير موجود. يرجى إنشاء التمثيلات أولاً.")

#     bert_data = joblib.load("bert_embeddings.joblib")
#     data = bert_data["data"]
#     model_path = bert_data.get("model_path")

#     if model_path is None or not os.path.exists(model_path):
#         raise ValueError("مسار النموذج غير موجود أو غير معرف داخل بيانات التمثيل.")

#     model = SentenceTransformer(model_path)

#     # query_embedding = model.encode(query, convert_to_tensor=True)
#     query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)

#     # doc_embeddings = [item["embedding"] for item in data]
#     # doc_embeddings_tensor = torch.tensor(doc_embeddings)
#     doc_embs = torch.tensor([item["embedding"] for item in data], dtype=query_embedding.dtype, device=query_embedding.device)

#     similarities = util.cos_sim(query_embedding, doc_embs)[0]

#     top_results = torch.topk(similarities, k=top_k)

#     results = []
#     for score, idx in zip(top_results.values, top_results.indices):
#         item = data[idx]
#         results.append({
#             "doc_id": data[idx]["doc_id"],
#             # "text": item["text"][:200] + "...",   
#             "score": round(score.item(), 4)
#         })

#     return results


def search(
    query: str,
    vector_path: str = "bert_embeddings.joblib",   
    model_dir: str = "bert_model",                 
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
    results = [
        {
            "doc_id": doc_ids[int(idx)],
            "score" : round(score.item(), 4)
        }
        for score, idx in zip(top_scores, top_indices)
    ]

    return results
