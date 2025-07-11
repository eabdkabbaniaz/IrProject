from fastapi import APIRouter
from models.models import HybirdRequest 
from controllers.viewHybirdSequentialController import viewHybirdSequentialController

router = APIRouter()
controller = viewHybirdSequentialController()

@router.post("/preview-HybirdSequential")
def hybird_sequential(data: HybirdRequest):
    result = controller.hybird_sequential(data.query, data.model_dir, data.vector_path, data.top_k)
    return {"data": result}
