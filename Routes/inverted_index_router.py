from fastapi import APIRouter
from models import IndexRequest
from controllers.inverted_index_controller import InvertedIndexController

router = APIRouter()
controller = InvertedIndexController()

@router.post("/generate-index")
def generate_index(request: IndexRequest):
    return controller.generate(request.input_path, request.output_path)
