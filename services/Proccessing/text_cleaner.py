import pandas as pd
import logging
from services.Proccessing.TextProcessing import TextProcessor, process_text  # استدعاء من ملفك الحالي

class TextCleaningService:
    def __init__(self):
        self.processor = TextProcessor()

    # def extract_doc_and_text(self, row):
    #     if len(row) == 1 and 'doc_id,text' in row and pd.notna(row['doc_id,text']):
    #         parts = row['doc_id,text'].split(',', 1)
    #         if len(parts) == 2:
    #             return parts[0].strip(), parts[1].strip().strip('"')
    #     # elif len(row) >= 2 and pd.notna(row[1]):
    #     #     return str(row[0]), str(row[1])
    #     elif len(row) >= 2 and pd.notna(row.iloc[1]):
    #         return str(row.iloc[0]), str(row.iloc[1])

    #     return None, None
    def extract_doc_and_text(self, row: pd.Series):
        logging.debug(f"extract_doc_and_text() => row.index={list(row.index)}")

        if len(row) == 1 and "doc_id,text" in row and pd.notna(row["doc_id,text"]):
            parts = row["doc_id,text"].split(",", 1)
            if len(parts) == 2:
                doc_id, text = parts[0].strip(), parts[1].strip().strip('"')
                logging.debug(f"Single‑column mode: doc_id={doc_id!r}")
                return doc_id, text

        elif len(row) >= 2 and pd.notna(row.iat[1]):      # استخدم iat/iloc لتفادي التحذير
            logging.debug(f"Two‑column mode: doc_id={row.iat[0]!r}")
            return str(row.iat[0]), str(row.iat[1])

        logging.debug("No valid doc/text found")
        return None, None

    def clean(self, df: pd.DataFrame):
        logging.debug("CleanerService.clean() => start")
        processed_rows = []
        for index, row in df.iterrows():
            logging.debug(f"Processing row {index}")

            try:
                doc_id, text = self.extract_doc_and_text(row)
                logging.debug(f"extract_doc_and_text => doc_id={doc_id!r}, text_preview={text[:30]!r}")

                if not text:
                    logging.debug("Empty text → skip row")
                    continue

                processed_text = process_text(text, self.processor)
                logging.debug(f"process_text => len={len(processed_text)}")
                logging.info(f"صف {index} - ID: {doc_id} - النص المعالج: {processed_text[:50]}...")

                if processed_text.strip():
                    processed_rows.append([doc_id, processed_text])
            except Exception as e:
                logging.error(f"خطأ في معالجة الصف {index}: {e}")
                continue
        logging.debug(f"CleanerService.clean() => processed_rows={len(processed_rows)}")

        return processed_rows
