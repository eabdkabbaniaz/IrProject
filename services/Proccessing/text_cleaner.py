import pandas as pd
import logging
from services.Proccessing.TextProcessing import TextProcessor, process_text  # استدعاء من ملفك الحالي

class TextCleaningService:
    def __init__(self):
        self.processor = TextProcessor()

    def extract_doc_and_text(self, row):
        if len(row) == 1 and 'doc_id,text' in row and pd.notna(row['doc_id,text']):
            parts = row['doc_id,text'].split(',', 1)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip().strip('"')
        elif len(row) >= 2 and pd.notna(row[1]):
            return str(row[0]), str(row[1])
        return None, None

    def clean(self, df: pd.DataFrame):
        processed_rows = []
        for index, row in df.iterrows():
            try:
                doc_id, text = self.extract_doc_and_text(row)
                if not text:
                    continue
                processed_text = process_text(text, self.processor)
                logging.info(f"صف {index} - ID: {doc_id} - النص المعالج: {processed_text[:50]}...")
                if processed_text.strip():
                    processed_rows.append([doc_id, processed_text])
            except Exception as e:
                logging.error(f"خطأ في معالجة الصف {index}: {e}")
                continue
        return processed_rows
