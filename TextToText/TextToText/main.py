# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app import app as fastapi_app  # Import your FastAPI app from app.py

# Create a new FastAPI app instance
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Mount your FastAPI app
app.mount("/api", fastapi_app)
