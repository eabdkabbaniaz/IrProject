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
