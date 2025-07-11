from fastapi import APIRouter
from models.models import ApplayInvertedIndex
from controllers.ApplayInvertedIndex import ApplayInvertedIndexController

router = APIRouter()
controller = ApplayInvertedIndexController()

@router.post("/applayInvertedIndex")
def run_applayInverted_service(data: ApplayInvertedIndex):
    result = controller.execute(
    data.corpus_path,
    data.output_index_path
    )
    return {"status": "success", "message": result}
