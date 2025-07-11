from fastapi import FastAPI
from Routes.query_suggest_routes import router as query_suggest_routes

app = FastAPI()
app.include_router(query_suggest_routes)
