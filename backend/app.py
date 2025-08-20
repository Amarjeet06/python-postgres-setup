from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
import os
import google.generativeai as genai

# Load env and configure Gemini
load_dotenv(find_dotenv())
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

# CORS (adjust origin if you want to lock it down)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # e.g. ["http://127.0.0.1:5500", "https://yourapp.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str

@app.get("/")
async def home():
    return {"message": "API is working!"}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not GEMINI_API_KEY:
        return {"response": "Server error: GEMINI_API_KEY is missing. Add it to .env and restart."}

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(request.user_input)

        # Gemini responses can be empty if blocked by safety filters
        text = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text
                                               if getattr(resp, "candidates", None) else "")
        if not text:
            text = "Sorry, I couldn’t generate a response."

        return {"response": text}

    except Exception as e:
        return {"response": f"Error: {str(e)}"}
