from fastapi import FastAPI
from src.routes import router

app = FastAPI(title="SecureContent AI", version="0.1")

app.include_router(router)