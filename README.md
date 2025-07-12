# IRProject 
An Information Retrieval System combining classical and modern approaches for efficient document search and ranking.

## Overview
This project is an IR (Information Retrieval) system designed to process, index, and retrieve documents based on user queries using traditional and modern techniques like BM25 and BERT. It supports a modular SOA-based architecture and provides core features like document cleaning, inverted indexing, tf-idf & BM25 scoring, and hybrid ranking strategies.

---

## Core Features

### 1. Document Cleaning
Cleans datasets/documents using:
- Tokenization
- Stemming
- Lemmatization
- And other preprocessing steps

### 2. Inverted Index
Builds an efficient JSON-based inverted index after cleaning. It serves as a fast-access backend for tf-idf and BM25 computations.

### 3. TF-IDF Matrix
Constructs a term-document tf-idf matrix using the inverted index to enable query-document matching via cosine similarity.

### 4. BERT Embedding
Trains and uses a BERT model to generate semantic embeddings of terms, capturing their contextual meaning for improved retrieval accuracy.

### 5. Sequential Hybrid Representation
Uses BM25 for initial document matching and BERT-based re-ranking to refine results based on semantic relevance.

### 6. Query Interface
Allows users to submit queries via a search box, select datasets, and apply the same cleaning procedures to the query to improve matching accuracy.

---

## Additional Features

### 1. BM25 Matrix
Provides BM25 score-based term weighting and document ranking — considers global corpus frequency for better importance calculation.

### 2. Branching Hybrid Representation
Combines BM25 and BERT outputs using Reciprocal Rank Fusion (RRF) to generate a unified ranked list of documents.

### 3. Smart Query Features
- Auto-complete query terms  
- Suggest alternative queries  
- Correct potentially misspelled queries  

---

## Project Architecture (SOA-based)

The system follows a layered **Service-Oriented Architecture (SOA)** with modular components:

- **API Layer:** Runs the FastAPI server via `uvicorn`, receives incoming requests on a designated port.
- **Routes Layer:** Maps endpoints (URLs and HTTP methods) to specific logic via controllers.
- **Models Layer:** Defines the structure, parameter names, and types expected in service calls.
- **Controllers Layer:** Routes incoming requests to the appropriate service.
- **Services Layer:** Contains the core logic and algorithm implementations for each feature.

---

## Datasets Supported

- **BEIR** (Benchmark suite) via `ir_datasets`
- **ANTIQUE** via `ir_datasets`

---

## Requirements

Make sure to install:
- Python 3.8+
- FastAPI
- Uvicorn
- Transformers
- scikit-learn
- ir_datasets  
(Full list in `requirements.txt`)

---

## Running the Project

```bash
# Step 1: Clone the repository
git clone -b eabd https://github.com/eabdkabbaniaz/IrProject.git
cd IrProject

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Start the API server
uvicorn Interface:app --port 8000

# Step 4: run main function in app.py
