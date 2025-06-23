from fastapi import FastAPI
from Routes.view_routes_parallel import router as view_router_parallel

app = FastAPI()
app.include_router(view_router_parallel)