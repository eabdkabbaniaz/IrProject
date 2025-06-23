from fastapi import FastAPI
from Routes.view_routes import router as view_router

app = FastAPI()
app.include_router(view_router)