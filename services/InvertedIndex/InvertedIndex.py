import json
import pandas as pd
from collections import defaultdict

def build_inverted_index(corpus_path, output_index_path):
    # قراءة ملف CSV
    df = pd.read_csv(corpus_path, sep='\t', encoding='utf-8')  # عدّل sep إذا كان ',' بدل '\t'

    inverted_index = defaultdict(set)

    # التأكد من وجود الأعمدة المطلوبة
    if 'ID' not in df.columns or 'Processed_Text' not in df.columns:
        raise ValueError("CSV must contain 'ID' and 'Processed_Text' columns")

    for _, row in df.iterrows():
        doc_id = str(row['ID'])
        text = str(row['Processed_Text'])

        for word in text.split():
            inverted_index[word].add(doc_id)  # استخدم set لمنع التكرار

    # تحويل sets إلى lists لأن JSON لا يدعم set
    index_as_dict = {word: list(doc_ids) for word, doc_ids in inverted_index.items()}

    # حفظ الملف بصيغة JSON
    with open(output_index_path, 'w', encoding='utf-8') as f:
        json.dump(index_as_dict, f, ensure_ascii=False, indent=2)

    print(f"Inverted index saved to {output_index_path}")
