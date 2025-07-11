import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from services.Proccessing.TextProcessing import TextProcessor

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

def run_search_engine(query, vectorizer_path, tfidf_matrix_path, corpus_path, original_texts_path, top_n=20):
    # تحميل الملفات
    vectorizer = joblib.load(vectorizer_path)
    tfidf_matrix = joblib.load(tfidf_matrix_path)
    df = pd.read_csv(corpus_path, delimiter='\t', header=None, names=['ID', 'Processed_Text'])
    # original_df = pd.read_csv(original_texts_path)
    original_df = pd.read_csv(original_texts_path, delimiter='\t')  # إذا كان الملف .tsv


    # التأكد من تطابق أنواع الـ ID
    df['ID'] = df['ID'].astype(str)
    original_df['doc_id'] = original_df['doc_id'].astype(str)

    # معالجة الاستعلام
    processed_query = process_text(query,processor)
    query_vec = vectorizer.transform([processed_query])

    # حساب التشابه مع كامل المصفوفة
    cos_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(cos_similarities)[::-1][:top_n]

    #print(f"\n🔍 أفضل {top_n} مستندات مطابقة للاستعلام: '{query}'\n")
    for rank, idx in enumerate(top_indices, start=1):
        doc_id = str(df.iloc[idx]['ID'])
        match = original_df[original_df['doc_id'] == doc_id]
        original_text = match.iloc[0]['text'] if not match.empty else "⚠️ النص الأصلي غير موجود."
        # return doc_id
        # print(f"{rank}. مستند رقم {idx} - ID: {doc_id} - التشابه: {cos_similarities[idx]:.4f}")
        # print(f"النص الأصلي: {original_text[:200]}...")
        # print("----")
    doc_ids = []
    for idx in top_indices:
        doc_id = str(df.iloc[idx]['ID'])
        doc_ids.append(doc_id)

    return doc_ids
