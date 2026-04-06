from fastapi import FastAPI
from src.routes import router

# Entry point for the FastAPI service
# This initializes the API and includes all routes

app = FastAPI(title="SecureContent AI", version="0.1")

# Register API routes (endpoints like /analyze)
app.include_router(router)