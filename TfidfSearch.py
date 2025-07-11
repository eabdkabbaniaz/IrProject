from fastapi import FastAPI
from Routes.ApplaySearchTfidfRoute import router as tfidfSearch

app = FastAPI()
app.include_router(tfidfSearch)