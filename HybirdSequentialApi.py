from fastapi import FastAPI
from Routes.view_routes_sequential import router as view_router_sequential

app = FastAPI()
app.include_router(view_router_sequential)