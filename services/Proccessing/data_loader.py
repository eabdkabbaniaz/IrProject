import os
import pandas as pd
import csv
from fastapi import HTTPException
import logging

class DataLoaderService:
    @staticmethod
    def load(input_path: str) -> pd.DataFrame:
        if not os.path.exists(input_path):
            raise HTTPException(status_code=400, detail="ملف الإدخال غير موجود")
        try:
            df = pd.read_csv(input_path, sep='\t', quoting=csv.QUOTE_ALL, on_bad_lines='warn')
            logging.info(f"تم تحميل الملف بنجاح: {input_path}، عدد الصفوف: {len(df)}")
            return df
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"فشل في قراءة الملف: {e}")
