import math
from collections import Counter
class TfIdfService:
    def compute_tfidf(documents, inverted_index, min_df=0.01, max_df=0.8):
        N = len(documents)
        idf = {}
        
        # حساب idf مع الفلترة
        for term, doc_ids in inverted_index.items():
            df = len(doc_ids)
            ratio = df / N
            if min_df <= ratio <= max_df:
                idf[term] = math.log((N / df) + 1)

        tfidf = {}

        for doc_id, tokens in documents.items():
            tfidf[doc_id] = {}
            term_counts = Counter(tokens)
            total_terms = len(tokens)

            for term, count in term_counts.items():
                if term in idf:  # فقط المصطلحات المقبولة بعد الفلترة
                    tf = count / total_terms
                    tfidf[doc_id][term] = tf * idf[term]

        return tfidf
