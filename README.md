My GPT (FastAPI + Gemini + HF)

Chat + Vision + Image Gen + PDF-QA + Personalized RAG + Answerability + Router + Next-Q + KPI Forecasts

This project is a full-stack AI assistant:

Chat (Gemini) with short-term memory

Vision (describe/analyze images)

Image Generation (Hugging Face FLUX by default)

PDF Reader (chunking + embeddings index per chat)

Personalized Retrieval & Reranking (embeddings → cross-encoder → click-learning boosts)

Answerability / Hallucination Risk (pre-answer risk scoring → fetch more / clarify)

Learned Intent Router (Chat vs PDF-QA vs Vision vs Image)

Next-Question Suggestions

KPI Ingest & Forecasts (CSV/XLSX/PDF tables → monthly series → SARIMAX or seasonal-naive)

Frontend: vanilla HTML/JS/CSS.
Backend: FastAPI + LangChain (Gemini) + SentenceTransformers + optional Transformers/torch + statsmodels.

✨ Features

Answerability Guardrail

Endpoint: /api/predict/answerability

Auto-wired before answering in /api/chat:

Low risk → normal answer

Medium risk → adds more retrieval

High risk → asks brief clarifying question

Personalized Retrieval & Reranking

Endpoint: /api/search (embeddings shortlist → CrossEncoder rerank → personal source boost)

Auto-RAG in the frontend: top 4–6 snippets injected with bracket citations [1].

👍/👎 source feedback calls /api/feedback/click (learns your preferred sources).

Intent Router

Endpoints: /api/route and /api/route/intent

Rules + zero-shot classifier (if transformers+torch available) to choose Chat / PDF-QA / Vision / Image.

Next-Question Suggestions

Endpoints: /api/next-questions, /api/predict/next_questions

KPI Forecasts

Ingest: /api/metrics/ingest (CSV/XLSX; PDF tables if pdfplumber available)

Forecast: /api/forecast/kpi (monthly resample → SARIMAX if available, else seasonal-naive)

🗂 Project Structure
python-postgres-setup/
├── backend/
│   └── app.py                # FastAPI server (all endpoints)
├── frontend/
│   ├── index.html            # Chat UI
│   ├── styles.css
│   ├── script.js             # RAG injection, feedback buttons, router, next-Q, image/pdf flows
│   ├── login.html
│   ├── login.js
│   ├── signup.html
│   └── signup.js
├── vercel.json               # Frontend rewrites (if using Vercel for static hosting)
├── .env                      # (create this) API keys & config
└── README.md

🔐 Environment Variables (.env)

Create a .env in the repo root:

# --- Auth / app ---
SECRET_KEY=change-this-in-prod
DATA_DIR=./data          # defaults to ./data; on Vercel, uses /tmp

# --- Google Gemini ---
GEMINI_API_KEY=your_google_api_key
GEMINI_TEXT_MODEL=gemini-2.0-flash
GEMINI_VISION_MODEL=gemini-2.5-flash
IMAGEN_MODEL=imagen-3.0-fast   # optional; HF used by default for image gen

# --- Hugging Face (image gen) ---
HF_TOKEN=your_hf_token
HF_IMAGE_MODEL=black-forest-labs/FLUX.1-schnell
IMAGE_PROVIDER=auto             # auto | hf | google

# --- Retrieval / Rerank / NLI ---
EMB_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=BAAI/bge-reranker-base
NLI_MODEL=microsoft/deberta-base-mnli
ROUTER_ZS_MODEL=facebook/bart-large-mnli

# --- Memory ---
CHAT_MEMORY_K=12


Notes

If torch/transformers aren’t installed, NLI and zero-shot router fallbacks are disabled automatically.

If statsmodels isn’t installed, KPI forecasting uses a seasonal-naive baseline.

▶️ Run Locally (Windows friendly)
1) Backend (FastAPI)
cd C:\Desktop\PythonProjects\python-postgres-setup

# (first time) allow venv activation in PowerShell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# create & activate venv
python -m venv venv
.\venv\Scripts\Activate

# install deps
pip install -r requirements.txt  # or pip install fastapi uvicorn python-dotenv langchain-google-genai sentence-transformers pypdf statsmodels pdfplumber huggingface_hub

# run server
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
# FastAPI live at http://127.0.0.1:8000

2) Frontend (Static)

Option A — open with a local server (recommended):

cd frontend
# Python http server
python -m http.server 5500
# open http://127.0.0.1:5500/index.html


Option B — VS Code “Live Server” extension and click “Go Live”.

The frontend auto-targets http://127.0.0.1:8000 in dev (and /api/* rewrites in prod/Vercel).

🔑 Auth (built-in)

Register: POST /auth/register → { username, password }

Login: POST /auth/login → { username, password }

Frontend stores JWT in localStorage and includes it as Authorization: Bearer <token>.

🧠 Core Endpoints (Cheatsheet)
Health
GET /

Chat (Answerability guardrail runs automatically)
POST /api/chat
{
  "user_input": "your question",
  "chat_id": "default"
}

Vision (describe an image)
POST /api/vision   # multipart/form-data
  image: <file>
  prompt: "Describe this image"
  chat_id: "default"

Image Generation (HF FLUX default)
POST /api/image/generate
{
  "prompt": "A cute corgi wearing sunglasses",
  "aspect_ratio": "1:1",
  "count": 1,
  "provider": "hf",          // or "auto" / "google"
  "chat_id": "default"
}

PDF Read (summarize or QA + index chunks per chat)
POST /api/pdf/read   # multipart/form-data
  pdf: <file>
  chat_id: "default"
  question: "What are key takeaways?"  # optional; omit to summarize

Personalized Retrieval & Reranking
POST /api/search
{
  "chat_id": "default",
  "query": "quarterly revenue growth",
  "k": 6,
  "pool": 18
}

Click-learning (👍/👎 on a source)
POST /api/feedback/click
{
  "chat_id": "default",
  "source": "Quarterly_Report_Q2.pdf",
  "positive": true
}

Answerability (standalone scoring)
POST /api/predict/answerability
{
  "question": "What was 2024 Q2 revenue?",
  "chat_id": "default",
  "k": 8
}

Intent Router
POST /api/route/intent
{
  "text": "Summarize the attached PDF",
  "has_pdf_index": true
}

Next-Question Suggestions
POST /api/next-questions
{
  "chat_id": "default",
  "n": 3
}

KPI Ingest & Forecasts
POST /api/metrics/ingest  # multipart/form-data
  file: <CSV/XLSX or PDF with tables>

POST /api/forecast/kpi
{
  "metric": "revenue",
  "horizon_months": 6,
  "groupby": ["region"],          // optional
  "filters": {"product": "Pro"}   // optional
}

🧩 How the Smart Bits Work

RAG Pipeline (Frontend)

On each send, the UI calls /api/search, takes top 4–6 reranked snippets, and injects them into the LLM prompt with a short instruction. The model is asked to cite with [1], [2] etc.

Answerability (Backend)

Computes coverage & keyword scores from retrieved chunks; optional NLI (entailment/contradiction) if transformers+torch installed.

Controls the chat flow: answer / retrieve more / ask to clarify.

Personalization

Every thumbs updates source_affinity per user. Scores are squashed to a small boost and included in ranking.

Router

Simple rules + zero-shot classifier (if available). Frontend exposes a “Route” button to test it.

KPI Forecasts

Ingest tabular metrics, resample to monthly, fit SARIMAX (if installed), fall back to seasonal-naive otherwise. Returns mean + band.

🛡 Data & Storage

Per-user chats and doc indices are stored under DATA_DIR (default ./data).

KPI data saved as parquet (if pyarrow present) else CSV.

On Vercel, filesystem is ephemeral—use an external volume or a database if you need persistence in production.

🚀 Deploy

Frontend: Any static host (Vercel, Netlify, S3). frontend/ contains static assets.

If using Vercel, keep vercel.json rewrites to proxy /api/* to your backend origin.

Backend: Deploy FastAPI with Uvicorn/Gunicorn on Render/Fly/EC2/Heroku-like platforms.
Set your .env securely in the host.

🧪 Quick Tests

Visit http://127.0.0.1:8000 → should return health JSON.

Login/register via the UI.

Upload a PDF → ask a question.

Click the Search button → see reranked results.

Send a question → answer should include bracket citations; feedback chips appear.

Try Route and Next-Q buttons.

Try Image Generate; you should see base64 images rendered inline.

🛠 Troubleshooting

PowerShell cannot activate venv
Run once: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

401 Unauthorized
Login again; the UI stores JWT in localStorage.

NLI / Router zero-shot disabled
Install transformers + torch:
pip install "transformers[torch]" (or torch per your CUDA/CPU)

No images returned (HF)
Check HF_TOKEN, and that the model is available to your account.

PDF didn’t index
Some PDFs are image-based; consider OCR or try another doc. For table extraction, pdfplumber helps.

🧱 Tech Stack

Backend: FastAPI, LangChain (Gemini), SentenceTransformers, optional Transformers/torch, statsmodels, pdfplumber, pypdf, huggingface_hub

Frontend: Vanilla HTML/CSS/JS, marked.js, highlight.js

Auth: JWT (jose, passlib/bcrypt)

Storage: JSON + parquet/CSV on disk
