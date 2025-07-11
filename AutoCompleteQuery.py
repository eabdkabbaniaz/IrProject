from fastapi import FastAPI
from Routes.autoCompleteQuery import router as autoComplete_routes

app = FastAPI()
app.include_router(autoComplete_routes)
