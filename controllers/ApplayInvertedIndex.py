from services.InvertedIndex.InvertedIndex import build_inverted_index

class ApplayInvertedIndexController:
    def execute(self, corpus_path, output_index_path):
        return build_inverted_index( corpus_path, output_index_path)
