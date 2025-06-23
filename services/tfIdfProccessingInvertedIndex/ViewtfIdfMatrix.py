import joblib
import pandas as pd
from itertools import islice

def viewMatrix(input_path, limit=100):
    tfidf_scores = joblib.load(input_path)
    subset = dict(islice(tfidf_scores.items(), limit))
    df = pd.DataFrame.from_dict(subset, orient='index').fillna(0)
    return df.reset_index().rename(columns={"index": "doc_id"}).to_dict(orient='records')
