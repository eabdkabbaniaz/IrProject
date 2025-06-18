import os
import pandas as pd
import json
from collections import defaultdict
from fastapi import HTTPException

class InvertedIndexController:
    def generate(self, input_path: str, output_path: str):
        if not os.path.exists(input_path):
            raise HTTPException(status_code=400, detail="Input file does not exist.")

        try:
            df = pd.read_csv(input_path, delimiter='\t', names=["id", "text"])

            inverted_index = defaultdict(set)

            for _, row in df.iterrows():
                doc_id = str(row['id'])
                text = str(row['text']).strip()
                words = text.split()
                for word in set(words):
                    inverted_index[word].add(doc_id)

            # تحويل القيم إلى قوائم
            inverted_index = {word: list(doc_ids) for word, doc_ids in inverted_index.items()}

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(inverted_index, f, ensure_ascii=False, indent=2)

            return {
                "message": "تم إنشاء الفهرس المعكوس بنجاح",
                "num_indexed_words": len(inverted_index),
                "output_path": output_path
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
