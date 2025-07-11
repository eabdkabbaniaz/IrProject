from fastapi import FastAPI
from Routes.processing_routes import router as clean_processing_service

app = FastAPI()
app.include_router(clean_processing_service)
