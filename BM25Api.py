from fastapi import FastAPI
from Routes.view_routes_BM25 import router as view_router_BM25

app = FastAPI()
app.include_router(view_router_BM25)