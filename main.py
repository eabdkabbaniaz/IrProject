from fastapi import FastAPI
from Routes.processing_routes import router as processing_router

app = FastAPI()
app.include_router(processing_router)
