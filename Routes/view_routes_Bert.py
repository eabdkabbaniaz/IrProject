from fastapi import APIRouter
from models import BertRequest , IndexRequest, SearchBertRequest
from controllers.viewBertMatrixController import viewBertMatrixController

router = APIRouter()
controller = viewBertMatrixController()

@router.post("/preview-Bert")
def preview_Bert(data: BertRequest):
    result = controller.bert_matrix(data.input_path ,data.model_path)
    return {"data": result}

@router.post("/preview-TrainModelBert")
def train_model_Bert(data: IndexRequest):
    result = controller.train_bert_model(data.input_path , data.output_path)
    return {"data": result}

@router.post("/preview-BertModelSearch")
def bert_model_search(data: SearchBertRequest):
    result = controller.search_bert_model(data.query,data.top_k)
    return {"data": result}
