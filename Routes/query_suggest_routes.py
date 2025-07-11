from fastapi import APIRouter
from models.models import QuerySuggest
from controllers.query_suggestController import QuerySuggestController

router = APIRouter()
controller = QuerySuggestController()

@router.post("/suggest_query")
def run_query_suggest_service(data: QuerySuggest):
    result = controller.execute(data.query,)
    return {"status": "success", "message": result}
