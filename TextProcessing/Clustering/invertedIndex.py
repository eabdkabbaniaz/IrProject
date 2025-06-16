import pandas as pd
from collections import defaultdict
import json

# 1. تحميل البيانات من الملف
file_path = r"C:\Users\vision\Desktop\IrProject\datasets\dataset_antic\resualt.csv"
df = pd.read_csv(file_path, delimiter='\t', names=["id", "text"])

# 2. بناء الفهرس المعكوس باستخدام defaultdict
inverted_index = defaultdict(set)

for _, row in df.iterrows():
    doc_id = str(row['id'])                      # تأكد أن المعرف نصي
    text = str(row['text']).strip()
    words = text.split()                         # تقسيم الكلمات (نص نظيف مسبقاً)

    for word in set(words):                      # استخدم set لتجنب التكرار
        inverted_index[word].add(doc_id)

# 3. تحويل القيم إلى قائمة (لأن JSON لا يدعم الـ set)
inverted_index = {word: list(doc_ids) for word, doc_ids in inverted_index.items()}

# 4. حفظ الفهرس المعكوس في ملف JSON
index_file = r"C:\Users\vision\Desktop\IrProject\datasets\dataset_antic\inverted_index.json"
with open(index_file, 'w', encoding='utf-8') as f:
    json.dump(inverted_index, f, ensure_ascii=False, indent=2)

print(f"✅ تم حفظ الفهرس المعكوس في: {index_file}")
print(f"🔢 عدد الكلمات المفهرسة: {len(inverted_index)}")
