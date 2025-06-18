from fastapi import FastAPI
from Routes.cleaning_router import router as cleaning_router
from Routes.tfidf_router import router as tfidf_router
from Routes.inverted_index_router import router as inverted_index_router

app = FastAPI()
app.include_router(cleaning_router, prefix="/clean", tags=["Cleaning"])
app.include_router(tfidf_router, prefix="/tfidf", tags=["TF-IDF"])
app.include_router(inverted_index_router, prefix="/index", tags=["Inverted Index"])

