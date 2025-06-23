from fastapi import APIRouter
from models import HybirdRequest 
from controllers.viewHybirdParallelController import viewHybirdParallelController

router = APIRouter()
controller = viewHybirdParallelController()

@router.post("/preview-HybirdParallel")
def hybird_parallel(data: HybirdRequest):
    result = controller.hybird_parallel(data.query,data.input_path, data.model_path, data.k1, data.b, data.top_k)
    return {"data": result}
