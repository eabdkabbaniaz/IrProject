from services.Proccessing.data_loader import DataLoaderService
from services.InvertedIndex import InvertedIndex          
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from collections import defaultdict , Counter
import pandas as pd
import numpy as np
import joblib
import os
import json
from services.Proccessing.TextProcessing import TextProcessor
from scipy.sparse import lil_matrix
from pathlib import Path
from typing import Any, Dict, List
from scipy.sparse import coo_matrix, save_npz
from tqdm.auto import tqdm

processor = TextProcessor()
with open(r"D:\IrProject\datasets\dataset_antic\stop_words.txt", 'r',encoding='utf-8') as file:
    words_to_remove = file.read().splitlines()
    
def process_text(text, processor):
    if text is None:
        return text

    text = processor.remove_html_tags(text)
    text = processor.normalize_unicode(text)
    text = processor.expand_contractions(text)

    text = processor.cleaned_text(text)
    text = processor.remove_urls(text)
    text = processor.remove_punctuation(text)
    text = processor.normalization_example(text)  

    text = processor.clean_text(text, words_to_remove)  
    text = processor.remove_stopwords(text)            

    text = processor.stemming_example(text)
    text = processor.lemmatization_example(text)
    text = processor.number_to_words(text)
    text = processor.handle_negations(text)

    return text

def build_bm25_matrix(
    documents: Any,
    inverted_index_path: str | Path,
    k1: float = 1.5,
    b: float = 0.75,
    output_matrix_path: str | Path | None = "bm25_matrix.npz",
) -> List[Dict[str, Any]]:
   
    # ---------------------------------------------------------------------
    # 1. Load and prepare the corpus (vectorised, no Python loop per row)
    # ---------------------------------------------------------------------
    print("🔄 Loading documents …")
    df = DataLoaderService.load(documents)[["ID", "Processed_Text"]]
    df = df.dropna(subset=["Processed_Text"]).reset_index(drop=True)
    print(f"✅ Loaded {len(df):,} documents with text.")

    # Tokenise each document once. ``str.split`` is *much* faster vectorised
    print("🪄 Tokenising corpus …")
    tokenized_corpus: List[List[str]] = df["Processed_Text"].str.split().tolist()
    pids: List[str] = df["ID"].astype(str).tolist()
    print("✅ Tokenisation done.")

    # ---------------------------------------------------------------------
    # 2. Fit BM25 model (fast Cython code inside rank_bm25)
    # ---------------------------------------------------------------------
    print("⚙️  Fitting BM25 model …")
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    print("✅ BM25 model ready.")

    # ---------------------------------------------------------------------
    # 3. Intersect vocabulary with inverted index keys (cheap set operation)
    # ---------------------------------------------------------------------
    print("📥 Loading inverted index …")
    with open(inverted_index_path, "r", encoding="utf-8") as fp:
        inverted_index = json.load(fp)
    print(f"✅ Inverted index loaded. Terms: {len(inverted_index):,}.")

    print("🔍 Building final vocabulary …")
    vocab = sorted(t for t in inverted_index.keys() if t in bm25.idf)
    print(f"✅ Vocabulary size: {len(vocab):,}.")
    term_to_col: Dict[str, int] = {t: j for j, t in enumerate(vocab)}

    # ---------------------------------------------------------------------
    # 4. Construct the sparse BM25 matrix in one pass over *documents*
    # ---------------------------------------------------------------------
    print("🏗️  Constructing sparse BM25 matrix …")
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    avgdl = bm25.avgdl  # small speed win for inner loop
    tqdm_iter = tqdm(
        enumerate(tokenized_corpus, start=0),
        total=len(tokenized_corpus),
        desc="📈 Scoring rows",
    )

    for i, tokens in tqdm_iter:
        len_doc = len(tokens)
        tf_counter = Counter(tokens)
        for term, tf in tf_counter.items():
            col = term_to_col.get(term)
            if col is None:
                continue  # term not in final vocab
            idf = bm25.idf[term]
            score = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len_doc / avgdl))
            if score:
                rows.append(i)
                cols.append(col)
                data.append(score)

    bm25_matrix = coo_matrix((data, (rows, cols)), shape=(len(pids), len(vocab))).tocsr()
    print(
        f"✅ Matrix built. Non‑zero entries: {bm25_matrix.nnz:,} (density={bm25_matrix.nnz/ (len(pids)*len(vocab)):.4%})."
    )

    # ---------------------------------------------------------------------
    # 5. Persist artefacts (matrix + model)
    # ---------------------------------------------------------------------
    if output_matrix_path is not None:
        print(f"💾 Saving matrix to {output_matrix_path} …")
        save_npz(output_matrix_path, bm25_matrix)
        print("✅ Sparse matrix saved.")

    print("💾 Saving BM25 model artefacts …")
    joblib.dump({"bm25": bm25, "pids": pids, "vocab": vocab}, "bm25_model.joblib")
    print("✅ BM25 model artefacts saved.")

    # ---------------------------------------------------------------------
    # 6. Stream cleaned_records from the sparse matrix (no dense DataFrame)
    # ---------------------------------------------------------------------
    print("🧹 Streaming cleaned records …")
    cleaned_records: List[Dict[str, float]] = []
    for i in range(bm25_matrix.shape[0]):
        row = bm25_matrix.getrow(i)
        indices = row.indices
        values = row.data
        record = {"doc_id": pids[i]}
        record.update({vocab[idx]: float(val) for idx, val in zip(indices, values)})
        cleaned_records.append(record)
    print("✅ Done. Returning cleaned_records list.")

    print("🎉 build_bm25_matrix finished successfully.")
    return cleaned_records
    
_bm25_tuple: tuple | None = None

def _load_bm25():
    global _bm25_tuple
    if _bm25_tuple is None:
        print("📦 Loading BM25 model into memory …")
        data = joblib.load("bm25_model.joblib")
        _bm25_tuple = data["bm25"], data["pids"], set(data["vocab"])
    return _bm25_tuple


def search(query: str, top_k: int = 25):
    bm25, pids, vocab = _load_bm25()

    query_tokens = process_text(query,processor)
    if isinstance(query_tokens, str):
        query_tokens = query_tokens.split()
    query_tokens = [t for t in query_tokens if t in vocab]

    top_k = min(top_k, len(pids))
    best_pids = bm25.get_top_n(query_tokens, pids, n=top_k)

    scores = bm25.get_scores(query_tokens)
    score_map = {pid: float(scores[pids.index(pid)]) for pid in best_pids}

    return [{"doc_id": pid, "score": round(score_map[pid], 4)} for pid in best_pids]