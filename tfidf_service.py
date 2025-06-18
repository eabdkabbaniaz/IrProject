# tfidf_service.py
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_tfidf(input_path: str, vectorizer_path: str, matrix_path: str):
    df = pd.read_csv(input_path, sep='\t')
    docs = df["text"].dropna().tolist()

    vectorizer = TfidfVectorizer(max_df=0.7, min_df=0.01)
    matrix = vectorizer.fit_transform(docs)

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(matrix, matrix_path)

    return {
        "vectorizer_saved_to": vectorizer_path,
        "matrix_saved_to": matrix_path,
        "matrix_shape": matrix.shape
    }
