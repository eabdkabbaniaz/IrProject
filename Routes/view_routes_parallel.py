from fastapi import APIRouter
from models.models import HybirdRequest 
from controllers.viewHybirdParallelController import viewHybirdParallelController

router = APIRouter()
controller = viewHybirdParallelController()

@router.post("/preview-HybirdParallel")
def hybird_parallel(data: HybirdRequest):
    result = controller.hybird_parallel(data.query, data.model_dir, data.vector_path, data.top_k)
    return {"data": result}
