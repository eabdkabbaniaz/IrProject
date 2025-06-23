import joblib
import logging
from fastapi import HTTPException

class TfidfSaverService:
    @staticmethod
    def save(tfidf_scores: dict, output_path: str):
        try:
            joblib.dump(tfidf_scores, output_path)
            logging.info(f"✅ تم حفظ مصفوفة TF-IDF باستخدام joblib في: {output_path}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"❌ فشل في حفظ مصفوفة TF-IDF: {e}")
