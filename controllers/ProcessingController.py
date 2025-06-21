from services.tfIdfProccessingInvertedIndex.tfIdfProccessingInvertedIndex import run_processing

class ProcessingController:
    def execute(self, input_path, output_path, inverted_index_path, tfidf_output_path):
        return run_processing(input_path, output_path, inverted_index_path, tfidf_output_path)
