import pandas as pd
from collections import Counter

# استخدام الفاصل tab
df = pd.read_csv(r"C:\Users\vision\Desktop\IrProject\datasets\dataset_antic\antique_documents_cleaned230.csv", sep='\t')

# تحويل كل وثيقة إلى قائمة كلمات
documents = {row["ID"]: row["Processed_Text"].split() for _, row in df.iterrows()}
print(documents["2020338_0"])  # لترى إن كانت "one" فعلاً موجودة

# مثال: عد مرات تكرار "one" في وثيقة محددة
print(Counter(documents["2020338_0"])["one"])
import json

with open(r"C:\Users\vision\Desktop\IrProject\datasets\dataset_antic\inverted_index1100.json", "r", encoding="utf-8") as f:
    inverted_index = json.load(f)

# مثال: كم وثيقة تحتوي على الكلمة "one"
print("عدد الوثائق التي تحتوي على 'one':", len(inverted_index.get("one", [])))


import math

# عدد الوثائق الكلي
N = len(documents)

# عدد الوثائق التي تحتوي على "one"
df = len(inverted_index.get("one", []))

# حساب الـ IDF
idf = math.log((N / df) + 1)

print(f"IDF للكلمة 'one' هو: {idf}")

# عدد كلمات الوثيقة
total_terms = len(documents["2020338_0"])

# TF = عدد مرات ظهور الكلمة / عدد كلمات الوثيقة
tf = 1 / total_terms

# IDF حسب ما حسبناه
idf = 2.032452486432758

# TF-IDF
tfidf = tf * idf

print(f"TF-IDF للكلمة 'one' في الوثيقة 2020338_0 هو: {tfidf:.6f}")


