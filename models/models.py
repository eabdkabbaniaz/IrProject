from pydantic import BaseModel

class CleanRequest(BaseModel):
    input_path: str
    output_path: str

class ApplayInvertedIndex(BaseModel):
    corpus_path:str
    output_index_path:str

class ApplayTfidf(BaseModel):
    corpus_path:str 
    vectorizer_output_path:str
    matrix_output_path:str
    terms_output_csv:str
    inverted_index_json_path:str

class SearchTfidf(BaseModel):
    query:str
    vectorizer_path:str 
    tfidf_matrix_path:str
    corpus_path:str
    original_texts_path:str
   
class BM25Request(BaseModel):
    input_path: str
    inverted_index_path:str
    k1:float=1.5
    b:float=0.75

class BertRequest(BaseModel):
    input_path: str
    output_path: str
    vector_path: str
  
class SearchBM25Request(BaseModel):
    query: str
    vector_path: str
    model_dir: str
    top_k: int

class SearchBertRequest(BaseModel):
    query: str
    vector_path: str
    model_dir: str
    top_k: int

class IndexRequest(BaseModel):
    input_path: str
    output_path: str

class QuerySuggest(BaseModel):
    query: str


class AutoCompleteQuery(BaseModel):
    query: str
        
class CorrectQuery(BaseModel):
    query: str

class HybirdRequest(BaseModel):
    query: str
    model_dir: str        
    vector_path: str
    top_k: int