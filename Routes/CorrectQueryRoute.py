from fastapi import APIRouter
from models.models import CorrectQuery
from controllers.CorrectQueryController import CorrectQueryController

router = APIRouter()
controller = CorrectQueryController()

@router.post("/correct_query")
def run_query_correct_service(data: CorrectQuery):
    result = controller.execute(
        data.query,
        
    )
    return {"status": "success", "message": result}
