from fastapi import APIRouter
from models import BM25Request , SearchBM25Request
from controllers.viewBM25MatrixController import viewBM25MatrixController

router = APIRouter()
controller = viewBM25MatrixController()

@router.post("/preview-BM25")
def preview_BM25(data: BM25Request):
    result = controller.bm25_matrix(data.input_path, data.inverted_index_path,data.k1,data.b)
    return {"data": result}

@router.post("/preview-BM25Search")
def BM25_search(data: SearchBM25Request):
    result = controller.bm25_search(data.query, data.k1, data.b, data.top_k)
    return {"data": result}
