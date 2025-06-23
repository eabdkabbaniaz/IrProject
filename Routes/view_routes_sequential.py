from fastapi import APIRouter
from models import HybirdRequest 
from controllers.viewHybirdSequentialController import viewHybirdSequentialController

router = APIRouter()
controller = viewHybirdSequentialController()

@router.post("/preview-HybirdSequential")
def hybird_sequential(data: HybirdRequest):
    result = controller.hybird_sequential(data.query,data.input_path, data.model_path, data.k1, data.b, data.top_k)
    return {"data": result}
