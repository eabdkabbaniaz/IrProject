import joblib
import numpy as np
import pandas as pd

# المسارات للملفات المحفوظة
vectorizer_path = r"C:\Users\vision\Desktop\IrProject\datasets\dataset_antic\tfidf_vectorizerAA.pkl"
matrix_path = r"C:\Users\vision\Desktop\IrProject\datasets\dataset_antic\tfidf_matrixAA.pkl"

# تحميل الـ vectorizer
vectorizer = joblib.load(vectorizer_path)
# تحميل مصفوفة TF-IDF
tfidf_matrix = joblib.load(matrix_path)

# الحصول على أسماء الكلمات (الأعمدة)
feature_names = vectorizer.get_feature_names_out()

# تحويل أول 5 مستندات إلى شكل كثيف (dense)
dense_matrix = tfidf_matrix[:5].toarray()

# إنشاء DataFrame لسهولة العرض
df_tfidf = pd.DataFrame(dense_matrix, columns=feature_names)

# عرض النتائج
print("عرض أول 5 مستندات كمصفوفة TF-IDF:")
print(df_tfidf)

# عرض أعلى الكلمات في أول مستند
row = tfidf_matrix[0].toarray().flatten()
top_indices = row.argsort()[::-1][:10]  # أعلى 10 كلمات
print("\nأعلى 10 كلمات في المستند الأول:")
for idx in top_indices:
    print(f"{feature_names[idx]}: {row[idx]}")
