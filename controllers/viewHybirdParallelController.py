from services.HybirdParallel.HybirdParallelService import hybird_parallel_represent

class viewHybirdParallelController:
    def hybird_parallel(self, query: str, input_path: str,model_path: str , k1: float, b: float , top_k: int):
        return hybird_parallel_represent(query, input_path, model_path, k1, b, top_k)