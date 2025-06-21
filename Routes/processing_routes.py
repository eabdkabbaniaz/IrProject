from fastapi import APIRouter
from models import ProcessingRequest
from Controllers.ProcessingController import ProcessingController

router = APIRouter()
controller = ProcessingController()

@router.post("/run-service")
def run_processing_service(data: ProcessingRequest):
    result = controller.execute(
        data.input_path,
        data.output_path,
        data.inverted_index_path,
        data.tfidf_output_path
    )
    return {"status": "success", "message": result}
