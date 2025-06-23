from services.Proccessing.data_loader import DataLoaderService
from services.Clustering.InvertedIndex import InvertedIndex
import pandas as pd
from collections import defaultdict
import numpy as np
import math
import joblib
import os

from nltk.tokenize import word_tokenize

def build_bm25_matrix(documents, inverted_index_path, k1=1.5, b=0.75):
    df = DataLoaderService.load(documents)

    documents = {row['ID']: row['Processed_Text'].split() for _, row in df.iterrows()}
    pids = list(documents.keys())

    inverted_index = InvertedIndex.build_inverted_index(documents)

    N = len(documents)
    avgdl = sum(len(doc) for doc in documents.values()) / N
    vocab = list(inverted_index.keys())

    term_freqs = {}
    doc_lengths = {}

    for pid, doc in documents.items():
        tf = defaultdict(int)
        for word in doc:
            tf[word] += 1
        term_freqs[pid] = tf
        doc_lengths[pid] = len(doc)

    bm25_matrix = np.zeros((N, len(vocab)))
    pid_to_index = {pid: i for i, pid in enumerate(pids)}

    for j, term in enumerate(vocab):
        df_term = len(inverted_index[term])
        idf = math.log((N - df_term + 0.5) / (df_term + 0.5) + 1)

        for pid in inverted_index[term]:
            tf = term_freqs[pid][term]
            dl = doc_lengths[pid]
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            score = idf * ((tf * (k1 + 1)) / denom)
            i = pid_to_index[pid]
            bm25_matrix[i][j] = score

    bm25_df = pd.DataFrame(bm25_matrix, columns=vocab)
    bm25_df.insert(0, "doc_id", pids)

    records = bm25_df.to_dict(orient="records")

    cleaned_records = []
    for record in records:
        filtered_record = {"doc_id": record["doc_id"]}
        for key, value in record.items():
            if key != "doc_id" and value != 0.0:
                filtered_record[key] = value
        cleaned_records.append(filtered_record)

    joblib.dump({"data": cleaned_records}, "bm25_matrix.joblib")

     # Save raw components for dynamic scoring later
    bm25_raw = {
        "term_freqs": term_freqs,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "idf": {
            term: math.log((N - len(inverted_index[term]) + 0.5) / (len(inverted_index[term]) + 0.5) + 1)
            for term in vocab
        }
    }
    joblib.dump(bm25_raw, "bm25_raw_data.joblib")

    return cleaned_records


def search(query, top_k=5):
    query_terms = word_tokenize(query.lower())
    matching_terms = [term for term in query_terms if term in bm25_df.columns]

    if not matching_terms:
        return []

    if k1 == default_k1 and b == default_b:
        scores = bm25_df[matching_terms].sum(axis=1)
        ranked = scores.sort_values(ascending=False)
        return [(pid, score) for pid, score in ranked.items()]

    bm25_raw = joblib.load("bm25_raw_data.joblib")
    tf = bm25_raw["term_freqs"]
    dl = bm25_raw["doc_lengths"]
    idf = bm25_raw["idf"]
    avgdl = bm25_raw["avgdl"]

    scores = {}
    for pid in pids:
        score = 0.0
        for term in matching_terms:
            f = tf[pid].get(term, 0)
            doc_len = dl[pid]
            term_idf = idf.get(term, 0)
            denom = f + k1 * (1 - b + b * doc_len / avgdl)
            score += term_idf * ((f * (k1 + 1)) / denom) if denom else 0
        scores[pid] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked

def search(query, k1=1.5, b=0.75, top_k=25):

    if not os.path.exists("bm25_raw_data.joblib"):
        raise FileNotFoundError("الملف 'bm25_raw_data.joblib' غير موجود. قم ببناء المصفوفة أولاً.")

    if not os.path.exists("bm25_matrix.joblib"):
        raise FileNotFoundError("الملف 'bm25_matrix.joblib' غير موجود. قم ببناء المصفوفة أولاً.")

    query_terms = word_tokenize(query.lower())

    raw_data = joblib.load("bm25_raw_data.joblib")
    matrix_data = joblib.load("bm25_matrix.joblib")["data"]

    term_freqs = raw_data["term_freqs"]
    doc_lengths = raw_data["doc_lengths"]
    avgdl = raw_data["avgdl"]
    idf = raw_data["idf"]
    vocab = list(idf.keys())
    pids = list(term_freqs.keys())

    matching_terms = [term for term in query_terms if term in vocab]
    if not matching_terms:
        return []

    scores = {}
    for pid in pids:
        score = 0.0
        doc_len = doc_lengths[pid]
        for term in matching_terms:
            tf = term_freqs[pid].get(term, 0)
            term_idf = idf.get(term, 0)
            denom = tf + k1 * (1 - b + b * doc_len / avgdl)
            score += term_idf * ((tf * (k1 + 1)) / denom) if denom != 0 else 0
        scores[pid] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [{"doc_id": pid, "score": round(score, 4)} for pid, score in ranked]
