from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# ✅ Enable CORS so frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or ["http://127.0.0.1:5500"]
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
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4o-mini"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request.user_input}
            ]
        )

        reply = response.choices[0].message.content
        return {"response": reply}

    except Exception as e:
        return {"response": f"Error: {str(e)}"}
