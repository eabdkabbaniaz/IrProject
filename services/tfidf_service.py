import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import logging
from fastapi import HTTPException
import time

BASE_DATA_DIR = r"C:\Users\vision\Desktop\IrProject\datasets\dataset_antic"

class TFIDFService:

    @staticmethod
    def safe_path(filename: str) -> str:
        safe_path = os.path.join(BASE_DATA_DIR, filename)
        if not os.path.commonpath([BASE_DATA_DIR, os.path.abspath(safe_path)]) == os.path.abspath(BASE_DATA_DIR):
            raise HTTPException(status_code=400, detail="Invalid filename or path")
        if not os.path.exists(safe_path):
            raise HTTPException(status_code=404, detail="File not found")
        return safe_path

    def generate_tfidf(self, input_path: str, vectorizer_output_path: str, matrix_output_path: str,
                       max_df: float = 0.7, min_df: float = 0.01):

        # تحقق من صحة المسارات
        input_path = self.safe_path(input_path)
        vectorizer_output_path = os.path.join(BASE_DATA_DIR, vectorizer_output_path)
        matrix_output_path = os.path.join(BASE_DATA_DIR, matrix_output_path)

        logging.info(f"تحميل الملف: {input_path}")
        df = pd.read_csv(input_path, delimiter='\t')

        if 'Processed_Text' not in df.columns:
            raise HTTPException(status_code=400, detail="الملف لا يحتوي على عمود 'Processed_Text'")

        documents = df['Processed_Text'].astype(str).str.strip()
        documents = documents[documents != '']

        if documents.empty:
            raise HTTPException(status_code=400, detail="لا يوجد أي نصوص صالحة لإنشاء TF-IDF")

        logging.info(f"عدد المستندات بعد التنظيف: {len(documents)}")
        logging.info("بدء إنشاء مصفوفة TF-IDF...")

        start_time = time.time()

        vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
        tfidf_matrix = vectorizer.fit_transform(documents.tolist())

        elapsed = time.time() - start_time

        logging.info(f"تم إنشاء مصفوفة TF-IDF بنجاح خلال {elapsed:.2f} ثانية")
        logging.info(f"شكل المصفوفة: {tfidf_matrix.shape}")

        joblib.dump(vectorizer, vectorizer_output_path)
        joblib.dump(tfidf_matrix, matrix_output_path)

        return {
            "message": "تم إنشاء مصفوفة TF-IDF بنجاح",
            "num_documents": len(documents),
            "matrix_shape": tfidf_matrix.shape,
            "time_taken_seconds": round(elapsed, 2)
        }
