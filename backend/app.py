# ==============================
# My GPT Backend (Vercel / HF-first)
# ==============================
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any
import os, json, uuid, time, base64

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv

# Auth
from jose import jwt, JWTError
from passlib.context import CryptContext

# LangChain (chat)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from starlette.concurrency import run_in_threadpool

# Optional Google GenAI (guarded so we don't crash if unset)
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# Hugging Face fallback (image gen)
try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None


# ------------------------ ENV & APP ------------------------
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
ALGO = "HS256"

# LLM choices (override in .env if you like)
TEXT_MODEL   = os.getenv("GEMINI_TEXT_MODEL",   "gemini-2.0-flash")
VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
IMAGEN_MODEL = os.getenv("IMAGEN_MODEL",        "imagen-3.0-fast")

# Provider switch: "auto" | "hf" | "google"
IMAGE_PROVIDER = (os.getenv("IMAGE_PROVIDER") or "auto").lower()

# Use /tmp on Vercel (read-only FS except /tmp). Falls back to ./data locally.
DATA_DIR = Path(
    os.getenv("DATA_DIR")
    or ("/tmp/mygpt-data" if os.getenv("VERCEL") or os.getenv("VERCEL_ENV") else "data")
)
DATA_DIR.mkdir(parents=True, exist_ok=True)


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer = HTTPBearer(auto_error=False)

app = FastAPI(title="My GPT Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # (for same-domain frontend on Vercel this is fine)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy Google client — only if key present and lib available
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GENAI_CLIENT = None
if GEMINI_API_KEY and genai is not None:
    try:
        GENAI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        # Do not crash the function; we can still use HF
        print("WARN: Google GenAI client init failed:", repr(e))
        GENAI_CLIENT = None


# ------------------------ MODEL DISCOVERY (Google) ------------------------
def list_model_names() -> List[str]:
    """Return model names visible to your Google API key (normalized)."""
    if GENAI_CLIENT is None:
        return []
    try:
        items = GENAI_CLIENT.models.list()
        names = []
        for m in (getattr(items, "models", None) or items):
            name = getattr(m, "name", None) or str(m)
            names.append(name.split("/")[-1])  # normalize
        return names
    except Exception as e:
        print("list_model_names error:", repr(e))
        return []

def pick_imagen_model() -> Optional[str]:
    """Pick the first Imagen model visible to your key (if any)."""
    for n in list_model_names():
        if "imagen" in n:
            return n
    return None


# ------------------------ SIMPLE USER STORE ------------------------
USERS_FILE = DATA_DIR / "users.json"

def load_users() -> Dict[str, Dict[str, Any]]:
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_users(users: Dict[str, Dict[str, Any]]):
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")

users_db = load_users()  # {username: {"password": <hashed>}}


# ------------------------ CHAT PERSISTENCE ------------------------
def user_file(username: str) -> Path:
    return DATA_DIR / f"{username}.json"

def load_user_chats(username: str) -> Dict[str, Any]:
    p = user_file(username)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"chats": {}}
    return {"chats": {}}

def save_user_chats(username: str, payload: Dict[str, Any]):
    user_file(username).write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ------------------------ AUTH HELPERS ------------------------
def hash_password(pw: str) -> str:
    return pwd_context.hash(pw)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_token(sub: str) -> str:
    payload = {"sub": sub, "exp": int(time.time()) + 60 * 60 * 24}  # 24h
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGO)

def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGO])
        return payload.get("sub")
    except JWTError as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from e

def auth_required(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> Dict[str, Any]:
    if not creds or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Not authenticated")
    username = decode_token(creds.credentials)
    if not username or username not in users_db:
        raise HTTPException(status_code=401, detail="Invalid user")
    return {"username": username}


# ------------------------ LANGCHAIN MEMORY (in-process) ------------------------
# One chain per (user, chat_id)
chains: Dict[str, ConversationChain] = {}

def chain_key(user: str, chat_id: str) -> str:
    return f"{user}:{chat_id}"

def get_chain(user: str, chat_id: str) -> ConversationChain:
    if not os.getenv("GEMINI_API_KEY"):
        # Make it explicit so failures are clear on Vercel
        raise HTTPException(status_code=500, detail="Server missing GEMINI_API_KEY for chat model.")
    key = chain_key(user, chat_id)
    if key not in chains:
        llm = ChatGoogleGenerativeAI(model=TEXT_MODEL, api_key=os.getenv("GEMINI_API_KEY"))
        memory = ConversationBufferWindowMemory(k=6, return_messages=True)
        chains[key] = ConversationChain(llm=llm, memory=memory, verbose=False)
    return chains[key]

def reset_chain(user: str, chat_id: str):
    key = chain_key(user, chat_id)
    if key in chains:
        llm = ChatGoogleGenerativeAI(model=TEXT_MODEL, api_key=os.getenv("GEMINI_API_KEY"))
        chains[key] = ConversationChain(llm=llm, memory=ConversationBufferWindowMemory(k=6, return_messages=True))


# ------------------------ Pydantic Models ------------------------
class RegisterReq(BaseModel):
    username: str
    password: str

class LoginReq(BaseModel):
    username: str
    password: str

class ChatReq(BaseModel):
    user_input: str
    chat_id: Optional[str] = "default"

class ImageGenReq(BaseModel):
    prompt: str
    aspect_ratio: Optional[str] = "1:1"
    count: int = 1
    provider: Optional[str] = None   # "auto" | "google" | "hf"
    model: Optional[str] = None      # e.g., "stabilityai/sd-turbo"


# ------------------------ ROUTES: HEALTH & MODELS ------------------------
@app.get("/")
def health():
    return {
        "ok": True,
        "text_model": TEXT_MODEL,
        "vision_model": VISION_MODEL,
        "image_model": IMAGEN_MODEL,
        "image_provider": IMAGE_PROVIDER,
        "google_enabled": GENAI_CLIENT is not None,
        "hf_token_present": bool(os.getenv("HF_TOKEN")),
    }

@app.get("/api/models")
def models(user=Depends(auth_required)):
    return {"models": list_model_names()}

@app.get("/api/image/status")
def image_status(user=Depends(auth_required)):
    return {
        "provider": IMAGE_PROVIDER,
        "imagen_env": IMAGEN_MODEL,
        "google_enabled": GENAI_CLIENT is not None,
        "google_visible_models": list_model_names(),
        "hf_token_present": bool(os.getenv("HF_TOKEN")),
        "hf_model": os.getenv("HF_IMAGE_MODEL"),
    }


# ------------------------ ROUTES: AUTH ------------------------
@app.post("/auth/register")
def register(req: RegisterReq):
    if req.username in users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password too short (min 6)")
    users_db[req.username] = {"password": hash_password(req.password)}
    save_users(users_db)
    token = create_token(req.username)
    return {"username": req.username, "token": token}

@app.post("/auth/login")
def login(req: LoginReq):
    user = users_db.get(req.username)
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(req.username)
    return {"username": req.username, "token": token}


# ------------------------ ROUTES: CHAT MGMT ------------------------
@app.get("/api/chats")
def list_chats(user=Depends(auth_required)):
    u = user["username"]
    store = load_user_chats(u)
    chats_meta = []
    for cid, c in store["chats"].items():
        chats_meta.append({
            "id": cid,
            "title": c.get("title") or "New chat",
            "updated_at": c.get("updated_at", int(time.time())),
        })
    chats_meta.sort(key=lambda x: x["updated_at"], reverse=True)
    return {"chats": chats_meta}

@app.post("/api/chats/new")
def new_chat(user=Depends(auth_required)):
    u = user["username"]
    store = load_user_chats(u)
    cid = uuid.uuid4().hex[:12]
    store["chats"][cid] = {"title": "New chat", "messages": [], "updated_at": int(time.time())}
    save_user_chats(u, store)
    return {"chat_id": cid}

@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str, user=Depends(auth_required)):
    u = user["username"]
    store = load_user_chats(u)
    chat = store["chats"].get(chat_id, {"messages": []})
    return {"messages": chat.get("messages", [])}

@app.post("/api/chats/{chat_id}/clear")
def clear_chat(chat_id: str, user=Depends(auth_required)):
    u = user["username"]
    store = load_user_chats(u)
    if chat_id in store["chats"]:
        store["chats"][chat_id]["messages"] = []
        store["chats"][chat_id]["updated_at"] = int(time.time())
        save_user_chats(u, store)
    reset_chain(u, chat_id)
    return {"ok": True}

@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, user=Depends(auth_required)):
    u = user["username"]
    store = load_user_chats(u)
    if chat_id in store["chats"]:
        del store["chats"][chat_id]
        save_user_chats(u, store)
    key = chain_key(u, chat_id)
    if key in chains:
        del chains[key]
    return {"ok": True}


# ------------------------ ROUTES: CHAT ------------------------
@app.post("/api/chat")
async def chat(req: ChatReq, user=Depends(auth_required)):
    u = user["username"]
    store = load_user_chats(u)

    chat_id = req.chat_id or "default"
    if chat_id not in store["chats"]:
        store["chats"][chat_id] = {"title": "New chat", "messages": [], "updated_at": int(time.time())}

    try:
        chain = get_chain(u, chat_id)
        reply = await run_in_threadpool(chain.predict, input=req.user_input)
    except HTTPException as he:
        # surface missing key etc
        raise he
    except Exception as e:
        return {"response": f"Error: {e}"}

    chat_obj = store["chats"][chat_id]
    now_str = time.strftime("%I:%M %p")
    chat_obj["messages"].append({"role": "user", "content": req.user_input, "time": now_str})
    chat_obj["messages"].append({"role": "assistant", "content": reply, "time": now_str})
    if chat_obj.get("title", "New chat") == "New chat":
        chat_obj["title"] = req.user_input[:40]
    chat_obj["updated_at"] = int(time.time())
    save_user_chats(u, store)

    return {"response": reply}

@app.post("/api/reset")
def reset_memory(chat_id: str = Form(...), user=Depends(auth_required)):
    u = user["username"]
    reset_chain(u, chat_id)
    return {"ok": True}


# ------------------------ ROUTES: VISION (image -> text) ------------------------
@app.post("/api/vision")
async def analyze_image(
    image: UploadFile = File(...),
    prompt: str = Form("Describe this image"),
    chat_id: str = Form("default"),
    user=Depends(auth_required),
):
    if GENAI_CLIENT is None or types is None:
        return JSONResponse(status_code=400, content={"detail": "Vision requires GEMINI_API_KEY to be set on the server."})
    try:
        img_bytes = await image.read()
        img_part = types.Part.from_bytes(
            data=img_bytes,
            mime_type=image.content_type or "image/jpeg",
        )
        result = GENAI_CLIENT.models.generate_content(
            model=VISION_MODEL,
            contents=[img_part, prompt],
        )
        text = getattr(result, "text", None) or "(no response)"

        # optional: persist to chat
        u = user["username"]
        store = load_user_chats(u)
        if chat_id not in store["chats"]:
            store["chats"][chat_id] = {"title": "New chat", "messages": [], "updated_at": int(time.time())}
        now_str = time.strftime("%I:%M %p")
        store["chats"][chat_id]["messages"].append({"role": "user", "content": f"(uploaded image) {prompt}", "time": now_str})
        store["chats"][chat_id]["messages"].append({"role": "assistant", "content": text, "time": now_str})
        store["chats"][chat_id]["updated_at"] = int(time.time())
        save_user_chats(u, store)

        return {"response": text}
    except Exception as e:
        print("Vision error:", repr(e))
        return JSONResponse(status_code=400, content={"detail": str(e)})


# ------------------------ HF HELPERS (text -> image) ------------------------
def get_hf_client(model_override: Optional[str] = None):
    """
    Return (client, model_name, note).
    If not configured properly, client is None and note explains why.
    """
    token = os.getenv("HF_TOKEN")
    model = model_override or os.getenv("HF_IMAGE_MODEL", "stabilityai/sd-turbo")
    if not InferenceClient:
        return None, model, "huggingface_hub not installed."
    if not token:
        return None, model, "HF_TOKEN is not set on server."
    try:
        client = InferenceClient(token=token)
        return client, model, None
    except Exception as e:
        return None, model, f"Failed to create HF client: {e}"

def generate_with_hf(prompt: str, count: int = 1, model_override: Optional[str] = None):
    """
    Generate images with Hugging Face, robust to hub versions.
    Returns (images[dataURL], used_model, note_or_None).
    """
    client, model, note = get_hf_client(model_override)
    if not client:
        return [], model, note

    n = max(1, min(count, 4))
    images: List[str] = []

    for _ in range(n):
        pil_img = None
        # Attempt 1: width/height explicit
        try:
            pil_img = client.text_to_image(
                prompt=prompt,
                model=model,
                width=512,
                height=512,
            )
        except TypeError as e:
            # If width/height not supported, try size string
            if "unexpected keyword argument 'width'" in str(e) or "unexpected keyword argument 'height'" in str(e):
                try:
                    pil_img = client.text_to_image(
                        prompt=prompt,
                        model=model,
                        size="512x512",
                    )
                except Exception as e2:
                    return [], model, f"Hugging Face error: {e2}"
            else:
                return [], model, f"Hugging Face error: {e}"
        except Exception as e:
            # Last resort: call with defaults only
            try:
                pil_img = client.text_to_image(prompt=prompt, model=model)
            except Exception as e2:
                return [], model, f"Hugging Face error: {e2}"

        if pil_img is None:
            return [], model, "Hugging Face returned no image."

        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        images.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8"))

    return images, model, None


# ------------------------ ROUTES: IMAGE GENERATION (text -> image) ------------------------
@app.post("/api/image/generate")
async def generate_image(req: ImageGenReq, user=Depends(auth_required)):
    """
    Provider priority:
      - If req.provider is set: honor it ("google" | "hf" | "auto")
      - Else use IMAGE_PROVIDER from .env (default "auto")
    Model overrides:
      - For Google: we try env IMAGEN_MODEL if visible; else first visible 'imagen*' model
      - For HF: req.model (if provided) overrides env HF_IMAGE_MODEL
    """
    provider = (req.provider or IMAGE_PROVIDER or "auto").lower()

    def _try_google(model_name: str):
        if GENAI_CLIENT is None or types is None:
            raise RuntimeError("Google Imagen not available (GEMINI_API_KEY not set or library missing).")
        cfg = types.GenerateImagesConfig(
            number_of_images=max(1, min(req.count, 4)),
            aspect_ratio=req.aspect_ratio or "1:1",
        )
        return GENAI_CLIENT.models.generate_images(
            model=model_name,
            prompt=req.prompt,
            config=cfg,
        )

    try:
        images: List[str] = []
        note: Optional[str] = None
        model_used: Optional[str] = None
        used_provider: Optional[str] = None

        # ------------ Google path ------------
        if provider in ("google", "auto"):
            visible = list_model_names() if GENAI_CLIENT is not None else []
            google_model = None
            if IMAGEN_MODEL and any(IMAGEN_MODEL in m for m in visible):
                google_model = IMAGEN_MODEL
            else:
                google_model = pick_imagen_model()

            if google_model:
                try:
                    model_used = google_model
                    resp = _try_google(model_used)

                    # Parse both possible shapes
                    if hasattr(resp, "images") and resp.images:
                        for img in resp.images:
                            raw = getattr(img, "bytes", None) or getattr(img, "image_bytes", None)
                            if raw:
                                images.append("data:image/png;base64," + base64.b64encode(raw).decode("utf-8"))

                    elif hasattr(resp, "generated_images") and resp.generated_images:
                        for gi in resp.generated_images:
                            blob = getattr(gi, "image", None)
                            raw = getattr(blob, "image_bytes", None) if blob else None
                            if raw:
                                images.append("data:image/png;base64," + base64.b64encode(raw).decode("utf-8"))

                    if hasattr(resp, "prompt_feedback") and resp.prompt_feedback:
                        note = str(resp.prompt_feedback)

                    if images:
                        used_provider = "google"
                except Exception as e:
                    print("Google Imagen error:", str(e))
            else:
                if provider == "google":
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "No Google Imagen model available to your API key."},
                    )

        # ------------ HF path ------------
        if not images and provider in ("auto", "hf"):
            imgs, hf_model, hf_note = generate_with_hf(req.prompt, req.count, model_override=req.model)
            if imgs:
                return {"images": imgs, "note": hf_note, "provider": "huggingface", "model_used": hf_model}
            else:
                return JSONResponse(
                    status_code=400,
                    content={"detail": hf_note or "No image returned by available providers."},
                )

        # If images from Google:
        if images:
            return {"images": images, "note": note, "provider": used_provider or "google", "model_used": model_used}

        return JSONResponse(status_code=400, content={"detail": "No image returned by available providers."})

    except Exception as e:
        print("Image generation error:", repr(e))
        return JSONResponse(status_code=400, content={"detail": str(e)})
