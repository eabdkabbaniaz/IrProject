from fastapi import APIRouter
from models.models import CleanRequest
from controllers.ProcessingController import ProcessingController

router = APIRouter()
controller = ProcessingController()

@router.post("/clean-service")
def clean_processing_service(data: CleanRequest):
    result = controller.execute(
        data.input_path,
        data.output_path,
    )
    return {"status": "success", "message": result}
