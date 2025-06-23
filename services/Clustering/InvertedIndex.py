from collections import defaultdict
class InvertedIndex:
 def build_inverted_index(documents):
     inverted_index = defaultdict(set)
     for doc_id, tokens in documents.items():
        for token in set(tokens):
            inverted_index[token].add(doc_id)
     return inverted_index