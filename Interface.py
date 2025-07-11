from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Routes.query_suggest_routes import router as query_suggest_routes
from Routes.autoCompleteQuery import router as auto_complete_routes
# from Routes.CorrectQueryRoute import router as correct_routes

from Routes.ApplaySearchTfidfRoute import router as tfidf_search_routes
from Routes.view_routes_Bert import router as bert_search_routes
from Routes.view_routes_sequential import router as sequential_search_routes

from Routes.view_routes_BM25 import router as bm25_search_routes
from Routes.view_routes_parallel import router as parallel_search_routes

app = FastAPI()

# السماح لـ Flask بالتواصل مع FastAPI
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"],  # أو ["http://127.0.0.1:5000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origins=["http://127.0.0.1:5000"]
)

# إضافة الراوتر
app.include_router(query_suggest_routes)
app.include_router(auto_complete_routes)
# app.include_router(correct_routes)

app.include_router(tfidf_search_routes)
app.include_router(bert_search_routes)
app.include_router(sequential_search_routes)

app.include_router(bm25_search_routes)
app.include_router(parallel_search_routes)