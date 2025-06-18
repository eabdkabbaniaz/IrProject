from services.tfidf_service import TFIDFService

class TFIDFController:
    def __init__(self):
        self.service = TFIDFService()

    def generate(self, input_path: str, vectorizer_output_path: str, matrix_output_path: str,
                 max_df: float = 0.7, min_df: float = 0.01):
        return self.service.generate_tfidf(input_path, vectorizer_output_path, matrix_output_path,
                                           max_df, min_df)
