from fastapi import FastAPI
from Routes.ApplayInvertedIndexRoute import router as InvertedIndex

app = FastAPI()
app.include_router(InvertedIndex)