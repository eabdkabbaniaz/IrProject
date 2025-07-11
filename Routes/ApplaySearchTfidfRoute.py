from fastapi import APIRouter
from models.models import SearchTfidf
from controllers.ApplaySearchTfidf import ApplayyTfidfSearchController

router = APIRouter()
controller = ApplayyTfidfSearchController()

@router.post("/applaySearchTfidf")
def run_applayTfidfSearch_service(data: SearchTfidf):
   
    result = controller.execute(
    data.query,
    data.vectorizer_path,
    data.tfidf_matrix_path,
    data.corpus_path,
    data.original_texts_path,
   
    )
    return {"status": "success", "message": result}
