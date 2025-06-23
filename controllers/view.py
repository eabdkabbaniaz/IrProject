import joblib
import pandas as pd
from itertools import islice

# تحميل المصفوفة من الملف
tfidf_scores = joblib.load(r'C:\Users\vision\Desktop\IrProject\datasets\dataset_antic\tf_idf.joblib')

# اختيار أول 100 مستند فقط (مثلاً)
subset = dict(islice(tfidf_scores.items(), 100))

# تحويلها إلى DataFrame
df = pd.DataFrame.from_dict(subset, orient='index').fillna(0)

# عرض الجدول
print(df)
