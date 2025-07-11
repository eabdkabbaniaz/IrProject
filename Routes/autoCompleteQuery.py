from fastapi import APIRouter
from models.models import AutoCompleteQuery
from controllers.AutoCompleteQuery import AutoCompletController

router = APIRouter()
controller = AutoCompletController()

@router.post("/autoComplete_query")
def run_query_suggest_service(data: AutoCompleteQuery):
    result = controller.execute(
        data.query,
        
    )
    return {"status": "success", "message": result}
