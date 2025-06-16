import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

corpus_file = r"C:\Users\vision\Desktop\IrProject\datasets\dataset_antic\resualt1.csv"
df = pd.read_csv(corpus_file, delimiter='\t', header=None, names=['id', 'text'])

df = df.dropna(subset=['text'])
print(f"Number of documents after dropping NaNs: {len(df)}")

documents = df['text'].tolist()
print(f"Number of documents: {len(documents)}")

vectorizer = TfidfVectorizer(max_df=0.7,min_df=0.01)

tfidf_matrix = vectorizer.fit_transform(documents)
print(f"Shape of TF-IDF matrix: {tfidf_matrix.shape}")

# Save the vectorizer
vectorizer_file = r"C:\Users\vision\Desktop\IrProject\datasets\dataset_antic\tfidf_vectorizerAA.pkl"
with open(vectorizer_file, 'wb') as file:
    joblib.dump(vectorizer, file)
print(f"TF-IDF vectorizer saved to {vectorizer_file}")

output_file = r"C:\Users\vision\Desktop\IrProject\datasets\dataset_antic\tfidf_matrixAA.pkl"
with open(output_file, 'wb') as file:
    joblib.dump(tfidf_matrix, file)
print(f"TF-IDF matrix saved to {output_file}")


import numpy as np
from scipy.sparse import load_npz, csr_matrix


def read_tfidf_npz(filepath):
    try:
        tfidf_matrix = load_npz(filepath)
        print(f"Successfully loaded TF-IDF matrix from {filepath}")
        print(f"Shape of the matrix: {tfidf_matrix.shape}")
        # You can inspect a small part of it
        # print(tfidf_matrix[:2, :10].toarray()) # Convert to dense array for viewing
        return tfidf_matrix
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the NPZ file: {e}")
        return None

# To get the vocabulary back, you'd usually save the vectorizer object itself
# import pickle
# def load_tfidf_vectorizer(filepath):
#     try:
#         with open(filepath, 'rb') as f:
#             vectorizer = pickle.load(f)
#         print(f"Successfully loaded TfidfVectorizer from {filepath}")
#         return vectorizer
#     except FileNotFoundError:
#         print(f"Error: Vectorizer file '{filepath}' not found.")
#         return None
#     except Exception as e:
#         print(f"An error occurred while loading the vectorizer: {e}")
#         return None

import joblib
def load_tfidf_vectorizer(filepath):
    try:
        vectorizer = joblib.load(filepath)
        print(f"Successfully loaded TfidfVectorizer from {filepath}")
        return vectorizer
    except FileNotFoundError:
        print(f"Error: Vectorizer file '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the vectorizer: {e}")
        return None

load_tfidf_vectorizer(vectorizer_file)