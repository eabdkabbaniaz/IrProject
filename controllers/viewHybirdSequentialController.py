from services.HybirdSequential.HybirdSequentialervice import hybrid_sequential_represent

class viewHybirdSequentialController:
    def hybird_sequential(self, query: str ,model_dir: str,vector_path: str, top_k: int):
        return hybrid_sequential_represent(query, model_dir,vector_path, top_k)