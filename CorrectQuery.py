from fastapi import FastAPI
from Routes.CorrectQueryRoute import router as query_correct_routes

app = FastAPI()
app.include_router(query_correct_routes)
