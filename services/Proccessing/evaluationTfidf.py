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
from services.Proccessing.TextProcessing import TextProcessor

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity


# تعريف معالج النصوص
processor = TextProcessor()
with open(r"D:\IrProject\datasets\dataset_antic\stop_words.txt", 'r',encoding='utf-8') as file:
    words_to_remove = file.read().splitlines()
    
def process_text(text, processor):
    if text is None:
        return text

    # 1. إزالة العلامات الغريبة والعناوين HTML
    text = processor.remove_html_tags(text)
    text = processor.normalize_unicode(text)
    text = processor.expand_contractions(text)

    # 2. تنظيف النص من الرموز وتحويله لصيغة موحدة
    text = processor.cleaned_text(text)
    text = processor.remove_urls(text)
    text = processor.remove_punctuation(text)
    text = processor.normalization_example(text)  # تحويل إلى lowercase

    # 3. تنظيف الكلمات
    text = processor.clean_text(text, words_to_remove)  # كلمات مخصصة
    text = processor.remove_stopwords(text)             # كلمات من nltk

    # 4. المعالجة اللغوية
    text = processor.stemming_example(text)
    text = processor.lemmatization_example(text)
    text = processor.number_to_words(text)
    text = processor.handle_negations(text)

    return text

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

# ==== دالة البحث ====

def run_search_engine(query, vectorizer_path, tfidf_matrix_path, corpus_path, original_texts_path, top_n=20):
    # تحميل الملفات
    vectorizer = joblib.load(vectorizer_path)
    tfidf_matrix = joblib.load(tfidf_matrix_path)
    df = pd.read_csv(corpus_path, delimiter='\t', header=None, names=['ID', 'Processed_Text'])
    # original_df = pd.read_csv(original_texts_path)
    original_df = pd.read_csv(original_texts_path, delimiter='\t')
    # التأكد من تطابق أنواع الـ ID
    df['ID'] = df['ID'].astype(str)
    original_df['doc_id'] = original_df['doc_id'].astype(str)

    # معالجة الاستعلام
    processed_query = process_text(query,processor)
    query_vec = vectorizer.transform([processed_query])

    # حساب التشابه مع كامل المصفوفة
    cos_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(cos_similarities)[::-1][:top_n]
    retrieved_doc_ids = []  # ← قائمة لتخزين المعرفات

    print(f"\n🔍 أفضل {top_n} مستندات مطابقة للاستعلام: '{query}'\n")
    for rank, idx in enumerate(top_indices, start=1):
        doc_id = str(df.iloc[idx]['ID'])
        retrieved_doc_ids.append(doc_id)  # ← نضيف المعرف للقائمة
    return retrieved_doc_ids
        # match = original_df[original_df['doc_id'] == doc_id]
        # original_text = match.iloc[0]['text'] if not match.empty else "⚠️ النص الأصلي غير موجود."

        # print(f"{rank}. مستند رقم {idx} - ID: {doc_id} - التشابه: {cos_similarities[idx]:.4f}")
        # print(f"النص الأصلي: {original_text[:200]}...")
        # print("----")
# def run_search_engine(query, vectorizer_path, tfidf_matrix_path, corpus_path, original_texts_path, top_n=20):
    vectorizer = joblib.load(vectorizer_path)
    tfidf_matrix = joblib.load(tfidf_matrix_path)
    df = pd.read_csv(corpus_path, delimiter='\t', header=None, names=['ID', 'Processed_Text'])

    df['ID'] = df['ID'].astype(str)
    original_df = pd.read_csv(original_texts_path)
    original_df['doc_id'] = original_df['doc_id'].astype(str)

    processed_query = process_text(query, processor)

    if not processed_query.strip():
        return []

    query_vec = vectorizer.transform([processed_query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # فلترة النتائج: خذ فقط التي لها درجة تشابه غير صفرية
    filtered = [(idx, score) for idx, score in enumerate(scores) if score > 0]
    filtered.sort(key=lambda x: x[1], reverse=True)

    top_indices = [idx for idx, _ in filtered[:top_n]]
    retrieved_doc_ids = [df.iloc[idx]['ID'] for idx in top_indices]

    return retrieved_doc_ids

# ==== دوال التقييم ====
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

# ==== التقييم الكامل ====
def evaluate_search_engine(queries_path, qrels_path, vectorizer_path, tfidf_matrix_path, corpus_path, original_texts_path, inverted_index_path, top_k=0):
    queries = load_queries_txt(queries_path)
    qrels = load_qrels_json(qrels_path)

    precisions, recalls, maps, mrrs = [], [], [], []

    for query in queries:
        query_id = query['query_id']
        query_text = query['query']
        retrieved_docs =  run_search_engine(query_text, vectorizer_path, tfidf_matrix_path, corpus_path, original_texts_path, top_n=top_k)

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

        print(f"🔍 Query {query_id}: P@{top_k}={p:.2f}, R@{top_k}={r:.2f}, AP={ap:.2f}, MRR={mrr:.2f}")
    print("✅ Processed query:", retrieved_docs)
    print("✅ Query terms:", relevant_docs)
    print("\n📊 === المتوسطات ===")
    print(f"✅ Mean Precision@{top_k}: {np.mean(precisions):.4f}")
    print(f"✅ Mean Recall@{top_k}: {np.mean(recalls):.4f}")
    print(f"✅ MAP: {np.mean(maps):.4f}")
    print(f"✅ MRR: {np.mean(mrrs):.4f}")

# ==== تشغيل رئيسي ====
if __name__ == "__main__":
    evaluate_search_engine(
        queries_path=r"D:\IrProject\datasets\dataset_webis-touche2020\queries.txt",
        qrels_path=r"D:\IrProject\datasets\dataset_webis-touche2020\qrels.json",
        vectorizer_path=r"D:\IrProject\datasets\dataset_webis-touche2020\tfidf_vectorizer_touche2020.joblib",
        tfidf_matrix_path=r"D:\IrProject\datasets\dataset_webis-touche2020\tfidf_matrix_touche2020.joblib",
        corpus_path=r"D:\IrProject\datasets\dataset_webis-touche2020\clean_docs.csv",
        original_texts_path=r"D:\IrProject\datasets\dataset_webis-touche2020\docs.tsv",
        inverted_index_path=r"D:\IrProject\datasets\dataset_webis-touche2020\Inverted_touche2020.json",
        top_k=10
    )