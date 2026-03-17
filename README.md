# AI PDF Summarizer & Q&A

A full-stack AI application for PDF summarization and question-answering using RAG (Retrieval-Augmented Generation) with Mistral-7B.

## Stack
- **Frontend**: HTML + CSS + Vanilla JS
- **Backend**: FastAPI (Python)
- **LLM**: Mistral-7B-Instruct-v0.2 via HuggingFace Inference API (free)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (local, free)
- **Vector Store**: FAISS (local, in-memory)
- **RAG Framework**: LangChain

## Project Structure
```
pdf-summarizer/
├── backend/
│   ├── main.py           # FastAPI routes
│   ├── rag.py            # RAG pipeline
│   ├── prompts.py        # Prompt templates
│   └── requirements.txt
├── frontend/
│   └── index.html        # Full UI
└── README.md
```

---

## Setup Instructions

### Step 1: Get a HuggingFace API Token (Free)
1. Go to https://huggingface.co and create a free account
2. Go to Settings → Access Tokens → New Token
3. Create a token with **Read** permission
4. Copy the token (starts with `hf_...`)

### Step 2: Set Up Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your HuggingFace token
export HF_TOKEN=hf_your_token_here    # Linux/Mac
set HF_TOKEN=hf_your_token_here       # Windows CMD
$env:HF_TOKEN="hf_your_token_here"   # Windows PowerShell

# Start the server
python main.py
```

The API will be live at: `http://localhost:8000`
API docs (auto-generated): `http://localhost:8000/docs`

### Step 3: Open Frontend

Simply open `frontend/index.html` in your browser. No build step required.

---

## How to Use

1. **Upload PDFs** — Drag & drop or click to select one or more PDF files
2. **Summarize** — Click "Summarize All" for an AI-generated structured summary
3. **Ask Questions** — Switch to Q&A Chat tab and ask anything about your documents
4. **Multi-PDF** — Upload multiple PDFs; the RAG engine searches across all of them
5. **Chat History** — The Q&A supports follow-up questions with conversation context

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/`      | Health check |
| POST   | `/upload` | Upload PDF files |
| POST   | `/summarize` | Summarize loaded PDFs |
| POST   | `/ask`   | Ask a question (RAG) |
| GET    | `/pdfs`  | List loaded PDFs |
| DELETE | `/pdfs/{id}` | Remove a PDF |
| DELETE | `/pdfs`  | Clear all PDFs |

---

## Configuration

Edit `backend/rag.py` to tune:
- `chunk_size` — Size of text chunks (default: 500)
- `chunk_overlap` — Overlap between chunks (default: 80)
- `k` — Number of chunks retrieved per query (default: 4)
- `temperature` — LLM creativity (default: 0.3, lower = more factual)
- `max_new_tokens` — Max response length (default: 1024)

Edit `backend/prompts.py` to customize:
- Summary format and structure
- Q&A behavior and instructions
- System persona

---

## Troubleshooting

**"API Offline" in the UI**
→ Make sure the FastAPI server is running on port 8000

**"Model is currently loading" error from HuggingFace**
→ Free tier models go to sleep. Wait 20-30 seconds and retry.

**Slow responses**
→ Free HuggingFace Inference API has rate limits. For production, use a local model with `llama-cpp-python` or upgrade to HF Pro.

**CORS errors**
→ The backend allows all origins by default. If deploying, restrict `allow_origins` in `main.py`.
