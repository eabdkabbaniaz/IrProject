from fastapi import APIRouter
from models import ViewRequest
from Controllers.viewTfIdfMatrixController import viewTfIdfMatrixController

router = APIRouter()
controller = viewTfIdfMatrixController()

@router.post("/preview-tfidf")
def preview_tfidf(data: ViewRequest):
    result = controller.preview(data.input_path, data.limit)
    return {"data": result}
