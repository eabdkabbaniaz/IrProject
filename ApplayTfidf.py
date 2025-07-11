from fastapi import FastAPI
from Routes.ApplayTfidfRoutes import router as tfidf

app = FastAPI()
app.include_router(tfidf)