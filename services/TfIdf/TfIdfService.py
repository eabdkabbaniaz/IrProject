
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# import joblib
# import numpy as np

# def process_and_save_tfidf(corpus_path, vectorizer_output_path, matrix_output_path, terms_output_csv, preview_csv=None):
#     # 1. قراءة الملف
#     df = pd.read_csv(corpus_path, delimiter='\t', header=None, names=['ID', 'Processed_Text'])

#     # 2. حذف السطور الفارغة أو NaN
#     df = df.dropna(subset=['Processed_Text'])
#     df = df[df['Processed_Text'].str.strip() != '']

#     # 3. تحويل النصوص لقائمة
#     documents = df['Processed_Text'].tolist()

#     # 4. إنشاء TF-IDF
#     vectorizer = TfidfVectorizer(
#         max_df=0.7, 
#         min_df=0.000017, 
#         preprocessor=None,
#         tokenizer=None,
#         lowercase=False,
#         stop_words=None,
#     )
#     tfidf_matrix = vectorizer.fit_transform(documents)

#     # 5. التأكد من تطابق عدد الصفوف مع عدد المستندات
#     assert tfidf_matrix.shape[0] == len(documents), "❌ عدد الصفوف لا يتطابق مع عدد الوثائق"

#     # 6. حفظ النموذج والمصفوفة
#     joblib.dump(vectorizer, vectorizer_output_path)
#     joblib.dump(tfidf_matrix, matrix_output_path)

#     # 7. حفظ التيرمات في CSV
#     feature_names = vectorizer.get_feature_names_out()
#     df_terms = pd.DataFrame(feature_names, columns=["term"])
#     df_terms.to_csv(terms_output_csv, index=False)

#     print(f"✅ عدد الصفوف في المصفوفة: {tfidf_matrix.shape[0]}")
#     print(f"✅ عدد المستندات الأصلية: {len(documents)}")
#     print(f"✅ تم استخراج {len(feature_names)} تيرم.")
#     print(f"📄 تم حفظ التيرمات في: {terms_output_csv}")

#     # 8. (اختياري) حفظ أول 5 صفوف كـ preview CSV
#     if preview_csv:
#         dense_matrix = tfidf_matrix[:5].todense()
#         df_preview = pd.DataFrame(dense_matrix, columns=feature_names)
#         df_preview.to_csv(preview_csv, index=False)
#         print(f"📄 تم حفظ أول 5 صفوف في: {preview_csv}")

#     return "Success tfidf"  # ترجيع النتائج لو احتجتها

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
import json

def process_and_save_tfidf(
    corpus_path,
    vectorizer_output_path,
    matrix_output_path,
    terms_output_csv,
    inverted_index_json_path,
    preview_csv=None,
      # <-- المسار إلى inverted index (اختياري)
):
    # 1. قراءة الملف
    df = pd.read_csv(corpus_path, delimiter='\t', header=None, names=['ID', 'Processed_Text'])

    # 2. حذف السطور الفارغة أو NaN
    df = df.dropna(subset=['Processed_Text'])
    df = df[df['Processed_Text'].str.strip() != '']

    # 3. تحويل النصوص لقائمة
    documents = df['Processed_Text'].tolist()

    # 4. تحميل المفردات من inverted index (إذا توفر)
    vocabulary = None
    if inverted_index_json_path:
        with open(inverted_index_json_path, 'r', encoding='utf-8') as f:
            inverted_index = json.load(f)
        vocabulary = list(inverted_index.keys())
        print(f"📥 تم تحميل {len(vocabulary)} مصطلح من inverted index.")

    # 5. إنشاء TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        vocabulary=vocabulary,  # <-- نمرر vocabulary من inverted index إن توفر
        max_df=0.7,
        min_df=0.000017,
        preprocessor=None,
        tokenizer=None,
        lowercase=False,
        stop_words=None,
    )

    # 6. بناء مصفوفة TF-IDF
    tfidf_matrix = vectorizer.fit_transform(documents)

    # 7. التأكد من تطابق عدد الصفوف مع عدد الوثائق
    assert tfidf_matrix.shape[0] == len(documents), "❌ عدد الصفوف لا يتطابق مع عدد الوثائق"

    # 8. حفظ النموذج والمصفوفة
    joblib.dump(vectorizer, vectorizer_output_path)
    joblib.dump(tfidf_matrix, matrix_output_path)

    # 9. حفظ التيرمات في CSV
    feature_names = vectorizer.get_feature_names_out()
    df_terms = pd.DataFrame(feature_names, columns=["term"])
    df_terms.to_csv(terms_output_csv, index=False)

    print(f"✅ عدد الصفوف في المصفوفة: {tfidf_matrix.shape[0]}")
    print(f"✅ عدد المستندات الأصلية: {len(documents)}")
    print(f"✅ تم استخراج {len(feature_names)} تيرم.")
    print(f"📄 تم حفظ التيرمات في: {terms_output_csv}")

    # 10. (اختياري) حفظ أول 5 صفوف كـ preview
    if preview_csv:
        dense_matrix = tfidf_matrix[:5].todense()
        df_preview = pd.DataFrame(dense_matrix, columns=feature_names)
        df_preview.to_csv(preview_csv, index=False)
        print(f"📄 تم حفظ أول 5 صفوف في: {preview_csv}")

    return "✅ Success TF-IDF"
