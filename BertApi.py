from fastapi import FastAPI
from Routes.view_routes_Bert import router as view_router_Bert

app = FastAPI()
app.include_router(view_router_Bert)