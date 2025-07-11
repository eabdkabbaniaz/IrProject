from fastapi import APIRouter
from models.models import ApplayTfidf
from controllers.ApplayTfidfController import ApplayTfidfController

router = APIRouter()
controller = ApplayTfidfController()

@router.post("/applayTfidf")
def run_applayTfidf_service(data: ApplayTfidf):
    result = controller.execute(
    data.corpus_path,
    data.vectorizer_output_path,
    data.matrix_output_path,
    data.terms_output_csv,
    data.inverted_index_json_path
    )
    return {"status": "success", "message": result}
