from fastapi import APIRouter
from models import CleanRequest
from controllers.cleaning_controller import CleaningController

router = APIRouter()
cleaning_controller = CleaningController()

@router.post("/")
def clean_text(request: CleanRequest):
    return cleaning_controller.clean_data(request.input_path, request.output_path)
