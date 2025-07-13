import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from Proccessing.TextProcessing import TextProcessor

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from evaluationTfidf import evaluate_search_engine
from evaluationBert import Bert_evaluate_search_engine
from evaluation_hybird_parallel_represent import evaluate_hybird_parallel_represent
from evaluation_hybrid_sequential_service import evaluate_hybrid_sequential_represent
from BM25_evaluation import BM25_evaluate_search_engine
 
# def print_comparison_table(results):
#     df = pd.DataFrame(results)
#     df = df.set_index("Model")
#       ("\n📊 جدول مقارنة الأداء:\n")
#       (df.round(4))
    
#     # عرض بصري إذا كنت داخل Jupyter
#     try:
#         from IPython.display import display
#         display(df.round(4))
#     except:
#         pass
def print_comparison_table(results, save_path="comparison_results.csv"):
    
    df = pd.DataFrame(results)
    df = df.set_index("Model")

    print("\n📊 جدول مقارنة الأداء:\n")
    print(df.round(4))

    # ✅ حفظ النتائج إلى ملف CSV
    df.to_csv(save_path)

    # عرض بصري داخل Jupyter (اختياري)
    try:
        from IPython.display import display
        display(df.round(4))
    except:
        pass



# ==== تشغيل رئيسي ====
if __name__ == "__main__":

    #      
    results = []
    results.append(evaluate_search_engine(
        queries_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\antique_train_queries.txt",
        qrels_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\antique_train_qrels.json",
        vectorizer_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\files\tfidf_vectorizer_ant.joblib",
        tfidf_matrix_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\files\tfidf_matrix_ant.joblib",
        corpus_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\antique_documents_clean.csv",
        original_texts_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\antique_documents.csv",
        inverted_index_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\InvertedIndexResult.json",
        top_k=10
    ))
    results.append(Bert_evaluate_search_engine(
        queries_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\antique_train_queries.txt",
        qrels_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\antique_train_qrels.json",
        top_k=10,
        bert_model=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\files\bert_model",
        embeddings=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\files\embeddings.joblib",
    ))

         

    results.append(evaluate_hybird_parallel_represent(
        bm25_model=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\files\bm25_model.joblib",
        queries_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\antique_train_queries.txt",
        qrels_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\antique_train_qrels.json",
        bert_model=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\files\bert_model",
        embeddings=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\files\embeddings.joblib",
        top_k=10
    ))
         

    results.append(evaluate_hybrid_sequential_represent(
        bm25_model=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\files\bm25_model.joblib",
        bert_model=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\files\bert_model",
        embeddings=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\files\embeddings.joblib",
        queries_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\antique_train_queries.txt",
        qrels_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\antique_train_qrels.json",
  
    ))
         

    results.append(BM25_evaluate_search_engine(
        queries_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\antique_train_queries.txt",
        bm25_model=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\files\bm25_model.joblib",
        qrels_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_antic\antique_train_qrels.json",
        top_k=10
    ))
    results2 = []
    results2.append(evaluate_search_engine(
        queries_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\query100.txt",
        qrels_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\qrels_quore.json",
        vectorizer_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\tfidf_vectorizer_quore.joblib",
        tfidf_matrix_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\tfidf_matrix_quore.joblib",
        corpus_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\docs_clean.csv",
        original_texts_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\docs.tsv",
        inverted_index_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\InvertedIndexResultQuore.json",
        top_k=10
    ))
    results2.append(Bert_evaluate_search_engine(
        queries_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\query100.txt",
        qrels_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\qrels_quore.json",
         top_k=10,
        bert_model=r"C:\Users\user\Downloads\IrProject\IrProject\bert_model",
        embeddings=r"C:\Users\user\Downloads\IrProject\IrProject\embeddings.joblib",
    ))

         

    results2.append(evaluate_hybird_parallel_represent(
        bm25_model=r"C:\Users\user\Downloads\IrProject\IrProject\bm25_model.joblib",
        queries_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\query100.txt",
        qrels_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\qrels_quore.json",
        bert_model=r"C:\Users\user\Downloads\IrProject\IrProject\bert_model",
        embeddings=r"C:\Users\user\Downloads\IrProject\IrProject\embeddings.joblib",        top_k=10
    ))
         
    results2.append(evaluate_hybrid_sequential_represent(
        bm25_model=r"C:\Users\user\Downloads\IrProject\IrProject\bm25_model.joblib",
        bert_model=r"C:\Users\user\Downloads\IrProject\IrProject\bert_model",
        embeddings=r"C:\Users\user\Downloads\IrProject\IrProject\embeddings.joblib",  
        queries_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\query100.txt",
        qrels_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\qrels_quore.json",
        top_k=10
    ))
         

    results2.append(BM25_evaluate_search_engine(
        queries_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\query100.txt",
        bm25_model=r"C:\Users\user\Downloads\IrProject\IrProject\bm25_model.joblib",
        qrels_path=r"C:\Users\user\Downloads\IrProject\IrProject\datasets\dataset_quore\qrels_quore.json",
        top_k=10
    ))
    print("dataset 1")

    print_comparison_table(results)
    print ("dataset 2")
    print_comparison_table(results2 ,save_path="comparison_results2.csv" )



