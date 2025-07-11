from services.Bert.BertService import train_bert_model, search

class viewBertMatrixController:
    def train_bert_model(self, input_path: str, output_path: str, vector_path: str):
        return train_bert_model(input_path , output_path, vector_path)

    def search_bert_model(self, query: str, vector_path: str, model_dir: str, top_k: int):
        return search(query, vector_path, model_dir,top_k)
