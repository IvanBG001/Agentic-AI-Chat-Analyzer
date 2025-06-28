from fastapi import FastAPI
from app.api.endpoints import router

app = FastAPI(title="AI Chat Analyzer")

app.include_router(router)

# Optional root
@app.get("/")
def read_root():
    return {"message": "Welcome to Chat Analyzer"}
