from services.Tfidf.TfIdfService import process_and_save_tfidf

class ApplayTfidfController:
    def execute(self, corpus_path, vectorizer_output_path, matrix_output_path, terms_output_csv,inverted_index_json_path, preview_csv=None):
        return process_and_save_tfidf( corpus_path, vectorizer_output_path, matrix_output_path, terms_output_csv,inverted_index_json_path)
