from services.HybirdSequential.HybirdSequentialervice import hybird_sequential_represent

class viewHybirdSequentialController:
    def hybird_sequential(self, query: str, input_path: str,k1: float, b: float ,model_path: str, top_k: int):
        return hybird_sequential_represent(query, input_path, model_path, k1, b, top_k)