from services.Proccessing.ServiceProccessing import ServiceProccessing
from services.TfIdf.TfIdfService import TfIdfService
from services.Clustering.inverted_index_saver import InvertedIndexSaver
from services.TfIdf.TfidfSaverService import TfidfSaverService
from services.Clustering.InvertedIndex import InvertedIndex

def run_processing(input_path, output_path, inverted_index_path, tfidf_output_path):
    serviceProccessing = ServiceProccessing()
    cleaned_data = serviceProccessing.clean_data(input_path, output_path)

    documents = {doc_id: text.split() for doc_id, text in cleaned_data}

    inverted_index = InvertedIndex.build_inverted_index(documents)
    InvertedIndexSaver.save(inverted_index, inverted_index_path)

    tfidf_scores = TfIdfService.compute_tfidf(documents, inverted_index)
    TfidfSaverService.save(tfidf_scores, tfidf_output_path)

    return "Processing and saving completed successfully"
