from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Root endpoint
@app.get("/")
async def home():
    return {"message": "API is working!"}

# Request model
class ChatRequest(BaseModel):
    user_input: str

# Chat endpoint
@app.post("/api/chat")
async def chat(request: ChatRequest):
    # Mock response instead of calling OpenAI
    answer = f"[Mock GPT Response] You said: {request.user_input}"
    return {"response": answer}
