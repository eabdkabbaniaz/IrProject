from fastapi import APIRouter
from models.models import BertRequest, SearchBertRequest
from controllers.viewBertMatrixController import viewBertMatrixController

router = APIRouter()
controller = viewBertMatrixController()

@router.post("/train-TrainModelBert")
def preview_Bert(data: BertRequest):
    result = controller.train_bert_model(data.input_path ,data.output_path, data.vector_path)
    return {"data": result}

@router.post("/preview-BertModelSearch")
def bert_model_search(data: SearchBertRequest):
    result = controller.search_bert_model(data.query,data.vector_path, data.model_dir ,data.top_k)
    return {"data": result}
