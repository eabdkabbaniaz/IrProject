from services.HybirdParallel.HybirdParallelService import hybird_parallel_represent

class viewHybirdParallelController:
    def hybird_parallel(self, query: str,model_dir: str,vector_path: str , top_k: int):
        return hybird_parallel_represent(query, model_dir, vector_path, top_k)