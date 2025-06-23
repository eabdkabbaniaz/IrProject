from services.Bert.BertService import build_bert_embeddings,train_bert_model, search

class viewBertMatrixController:
    def bert_matrix(self, input_path: str, model_path: str):
        return build_bert_embeddings(input_path , model_path)

    def train_bert_model(self, input_path: str ,output_path: str):
        return train_bert_model(input_path , output_path)

    def search_bert_model(self, query: str, top_k: int):
        return search(query,top_k)
