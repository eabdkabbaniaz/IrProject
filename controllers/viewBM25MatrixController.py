from services.BM25.BM25Service import build_bm25_matrix , search

class viewBM25MatrixController:
    def bm25_matrix(self, input_path: str, inverted_index_path: str , k1: float, b:float):
        return build_bm25_matrix(input_path, inverted_index_path, k1, b)

    def bm25_search(self, query: str ,k1: float ,b: float,top_k: int):
        return search(query ,k1 ,b ,top_k)