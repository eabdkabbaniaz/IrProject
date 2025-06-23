from pydantic import BaseModel

class CleanRequest(BaseModel):
    input_path: str
    output_path: str


class TFIDFRequest(BaseModel):
    input_path: str
    vectorizer_output_path: str
    matrix_output_path: str
    max_df: float = 0.7
    min_df: float = 0.01


class TFIDFPreviewRequest(BaseModel):
    vectorizer_path: str
    matrix_path: str
    cleaned_path: str
    preview_rows: int = 5


class IndexRequest(BaseModel):
    input_path: str
    output_path: str

class ProcessingRequest(BaseModel):
    input_path: str
    output_path: str
    inverted_index_path: str
    tfidf_output_path: str
    
class ViewRequest(BaseModel):
    input_path: str
    limit: int = 100

class BM25Request(BaseModel):
    input_path: str
    inverted_index_path:str
    k1:float=1.5
    b:float=0.75

class BertRequest(BaseModel):
    input_path: str
    model_path: str
  
class SearchBM25Request(BaseModel):
    query: str
    k1: float
    b: float
    top_k: int

class SearchBertRequest(BaseModel):
    query: str
    top_k: int