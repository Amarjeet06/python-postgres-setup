🤖 My GPT – FastAPI + Vercel

A lightweight chat + vision web app built with FastAPI and a simple vanilla JS frontend. It supports JWT login, chat memory, and basic image analysis. Designed to be small, readable, and easy to deploy on Vercel.

✨ What it does

Sign up / Log in (JWT, bcrypt)

Chat powered by Google Gemini with short-term memory per chat

Vision: upload an image and get a description/extract via Gemini Vision

Image Generation: under development (HF fallback wired but not fully stable in production)

Storage: JSON files locally (data/) or /tmp on Vercel

🧰 Stack

Backend: FastAPI, Uvicorn, python-jose, passlib
LLM/Vision: google-genai, langchain-google-genai
Images: huggingface_hub (fallback), pillow
Frontend: HTML/CSS/JS (no framework)
Deploy: Vercel (Python serverless) + static frontend

🚀 Quick start (local)
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# .env (minimal)
# SECRET_KEY=change-me
# GEMINI_API_KEY=your-google-ai-key
# IMAGE_PROVIDER=hf        # optional
# HF_TOKEN=hf_************ # needed if IMAGE_PROVIDER=hf
uvicorn backend.app:app --reload --port 8000


Open frontend/index.html (or serve the folder) and log in.

☁️ Deploy (Vercel)

api/index.py exposes the FastAPI app under /api/*

vercel.json routes /api/*, /auth/*, /docs, plus serves /frontend/*

Set env vars in Vercel (remember: DATA_DIR=/tmp/mygpt-data)

⚠️ Status of Image Generation

The text-to-image endpoint is present but still under development and may return errors depending on provider/model credentials.

👨‍💻 Developed by: Amarjeet Singh Minhas, under guidence of Amanjit
📁 Technologies: FastAPI, Python, Google GenAI, LangChain, Vercel
