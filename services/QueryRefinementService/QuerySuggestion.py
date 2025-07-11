import nltk
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))
from nltk.corpus import wordnet
from nltk.wsd import lesk
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from services.Proccessing.TextProcessing import TextProcessor, process_text
import json
import textdistance
import numpy as np
# تأكد من تحميل الموارد عند الحاجة
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# كلمات يجب تجاهلها أثناء التوسيع
EXCLUDED_LEMMAS = {
    "do", "does", "did", "make", "thing", "something", "someone", "entity", "object",
    "give", "have", "get", "be", "take", "person", "individual", "go",
    "i", "we", "you", "he", "she", "it", "they", "how", "to", "me", "can", "tell",
    "why", "where", "what", "which", "whom", "whose", "when", "will", "would", "should", "could",
    "may", "might", "shall", "must", "is", "am", "are", "was", "were", "been", "being"
}

# تحميل كلمات التوقف
with open(r"D:\IrProject\datasets\dataset_antic\stop_words.txt", 'r', encoding='utf-8') as file:
    stop_words = set(file.read().splitlines())

# إعداد معالج النصوص
processor = TextProcessor()

# ملفات الإدخال
query_file = r"D:\IrProject\datasets\dataset_quore\queries_quore.txt"
search_file = r"D:\IrProject\datasets\dataset_quore\queries_quore_refinedFinal26.jsonl"

# قراءة الاستعلامات الأصلية
original_queries = []
query_ids = []
processed_queries = []

with open(query_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            try:
                query_id, query = line.strip().split('\t', 1)
                original_queries.append(query)
                query_ids.append(query_id)
                processed_queries.append(process_text(query, processor))
            except ValueError as e:
                print(f"خطأ في السطر: {line}\n{e}")

# قراءة الاستعلامات الموسعة من الملف
search_queries = {}
with open(search_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            try:
                data = json.loads(line)
                search_queries[data["qid"]] = data["query"]
            except json.JSONDecodeError as e:
                print(f"خطأ في قراءة JSON: {line}\n{e}")

# معالجة الاستعلامات الموسعة وبناء نموذج TF-IDF
refined_texts = [
    process_text(search_queries[qid], processor)
    for qid in query_ids if qid in search_queries
]
vectorizer = TfidfVectorizer(max_df=0.85, min_df=2)
query_vectors = vectorizer.fit_transform(refined_texts)

# دالة التوسيع السياقي باستخدام lesk وWordNet
def refine_query(query):
    tokens = word_tokenize(query)
    pos_tags = pos_tag(tokens)
    refined_terms = set()

    for word, tag in pos_tags:
        word_lower = word.lower()
        if word_lower in EXCLUDED_LEMMAS or word_lower in stop_words:
            continue

        # استخدام lesk للمعنى السياقي
        synset = lesk(tokens, word)
        if synset:
            for lemma in synset.lemmas():
                lemma_name = lemma.name().replace('_', ' ').lower()
                if lemma_name not in EXCLUDED_LEMMAS and lemma_name != word_lower:
                    refined_terms.add(lemma_name)

        refined_terms.add(word_lower)

    return ' '.join(refined_terms)

# اقتراح الاستعلامات المشابهة
def suggest_similar_queries(query, n=10):
    refined_query = refine_query(query)
    processed_query = process_text(refined_query, processor)
    query_vector = vectorizer.transform([processed_query])
    cosine_similarities = cosine_similarity(query_vector, query_vectors)[0]

    # Levenshtein similarity
    processed_original = process_text(query, processor)
    levenshtein_similarities = [
        textdistance.levenshtein.normalized_similarity(
            processed_original,
            process_text(original_queries[i], processor)
        ) for i in range(len(original_queries))
    ]

    # مزيج من Cosine وLevenshtein
    combined_scores = 0.7 * cosine_similarities + 0.3 * np.array(levenshtein_similarities)
    top_indices = combined_scores.argsort()[-n:][::-1]

    # عرض النتائج
    print("\n" + "="*50)
    print(f"الاستعلام الأصلي: '{query}'")
    print(f"الاستعلام المحسن: '{refined_query}'")
    print("="*50 + "\n")
    results = []
    for i, idx in enumerate(top_indices, 1):
        print(f"{i}. [ID: {query_ids[idx]}] (Score: {combined_scores[idx]:.3f})")
        print(f"   النص: {original_queries[idx]}")
        print("-"*40)
        # results.append((original_queries[idx], query_ids[idx], combined_scores[idx]))
        results.append((original_queries[idx]))

    return results

# test_query = "How do I get college money?"

# test_query = "why do I have lower stomache pain?"
# test_query = "Why doesn't vodka freeze, or if it does, at what temperature??"
# test_query = "Why do illegal immigrants get blamed for chaning the national anthem to spanish?"
# test_query = "How do i stop myself from sneezing?"
# test_query = "why dont people like sheep?"
# test_query = "how should a pregnant woman deal with morning sickness?"

# test_query = "What can I do for more energy?"
# test_query = " Why do we always close our eyes when we sneeze?"
# _ = suggest_similar_queries(test_query)