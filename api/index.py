# api/index.py
# Vercel serverless entry point – exposes your FastAPI app to /api/*
from backend.app import app  # FastAPI instance defined in backend/app.py
