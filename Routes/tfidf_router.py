from fastapi import APIRouter
from models import TFIDFRequest
from controllers.tfidf_controller import TFIDFController
from models import TFIDFPreviewRequest
from services.tfIdfViewService import tfidf_preview_service

router = APIRouter()
controller = TFIDFController()

@router.post("/generate-tfidf")
def generate_tfidf(data: TFIDFRequest):
    return controller.generate(data.input_path, data.vectorizer_output_path, data.matrix_output_path,
                               data.max_df, data.min_df)






@router.post("/preview")
def tfidf_preview(request: TFIDFPreviewRequest):
    return tfidf_preview_service(request)