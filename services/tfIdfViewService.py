import joblib
import pandas as pd
from fastapi import HTTPException

def tfidf_preview_service(data):
    try:
        # تحميل البيانات
        vectorizer = joblib.load(data.vectorizer_path)
        tfidf_matrix = joblib.load(data.matrix_path)
        cleaned_df = pd.read_csv(data.cleaned_path, sep="\t")

        # تنظيف النصوص الفارغة
        cleaned_df['Processed_Text'] = cleaned_df['Processed_Text'].astype(str).str.strip()
        valid_indices = cleaned_df[cleaned_df['Processed_Text'] != ''].index
        cleaned_df_clean = cleaned_df.loc[valid_indices].reset_index(drop=True)

        if len(cleaned_df_clean) != tfidf_matrix.shape[0]:
            raise HTTPException(status_code=400, detail="عدد الوثائق لا يطابق عدد الصفوف في مصفوفة TF-IDF")

        terms = vectorizer.get_feature_names_out()
        preview_matrix = tfidf_matrix[:data.preview_rows].toarray()
        preview_df = pd.DataFrame(preview_matrix, columns=terms)
        preview_df.insert(0, "ID", cleaned_df_clean['ID'].iloc[:data.preview_rows].values)

        return {"data_preview": preview_df.to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
