from services.Tfidf.TfidfSearch import run_search_engine

class ApplayyTfidfSearchController:
    def execute(self,query, vectorizer_path, tfidf_matrix_path, corpus_path, original_texts_path):
        return run_search_engine( query, vectorizer_path, tfidf_matrix_path, corpus_path, original_texts_path)
