import pandas as pd
from fastapi import HTTPException
import logging

class DataSaverService:
    @staticmethod
    def save(processed_rows: list, output_path: str):
        try:
            df_out = pd.DataFrame(processed_rows, columns=['ID', 'Processed_Text'])
            df_out.to_csv(output_path, sep='\t', index=False)
            logging.info(f"تم حفظ النتائج في: {output_path}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"فشل في حفظ الملف: {e}")
