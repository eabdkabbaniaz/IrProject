import json
import logging
from fastapi import HTTPException

class InvertedIndexSaver:
    @staticmethod
    def save(inverted_index: dict, output_path: str):
        try:
            # نحول الـ set إلى list لأن JSON لا يدعم set
            serializable_index = {term: list(doc_ids) for term, doc_ids in inverted_index.items()}
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_index, f, ensure_ascii=False, indent=2)

            logging.info(f"✅ تم حفظ الفهرس المعكوس في: {output_path}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"❌ فشل في حفظ الفهرس: {e}")
