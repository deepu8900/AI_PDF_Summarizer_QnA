from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import uuid

from backend.rag import RAGPipeline

app = FastAPI(title="PDF Summarizer & QnA API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


rag = RAGPipeline()


class QuestionRequest(BaseModel):
    question: str
    chat_history: Optional[List[dict]] = []


class SummarizeRequest(BaseModel):
    pdf_ids: Optional[List[str]] = None


@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/upload")
async def upload_pdf(files: List[UploadFile] = File(...)):
    uploaded = []
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")
        pdf_id = str(uuid.uuid4())[:8]
        save_path = os.path.join(UPLOAD_DIR, f"{pdf_id}_{file.filename}")
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        chunk_count = rag.ingest_pdf(save_path, pdf_id, file.filename)
        uploaded.append({"pdf_id": pdf_id, "filename": file.filename, "chunks": chunk_count, "status": "ingested"})
    return {"uploaded": uploaded, "total_documents": len(rag.get_loaded_pdfs())}


@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    if not rag.get_loaded_pdfs():
        raise HTTPException(status_code=400, detail="No PDFs loaded. Please upload PDFs first.")
    summary = rag.summarize(pdf_ids=req.pdf_ids)
    return {"summary": summary, "pdf_ids": req.pdf_ids or list(rag.get_loaded_pdfs().keys())}


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    if not rag.get_loaded_pdfs():
        raise HTTPException(status_code=400, detail="No PDFs loaded. Please upload PDFs first.")
    answer, sources = rag.answer_question(req.question, req.chat_history)
    return {"answer": answer, "sources": sources, "question": req.question}


@app.get("/pdfs")
def list_pdfs():
    return {"pdfs": rag.get_loaded_pdfs()}


@app.delete("/pdfs/{pdf_id}")
def delete_pdf(pdf_id: str):
    if not rag.remove_pdf(pdf_id):
        raise HTTPException(status_code=404, detail="PDF not found")
    return {"message": f"PDF {pdf_id} removed"}


@app.delete("/pdfs")
def clear_all_pdfs():
    rag.reset()
    for f in os.listdir(UPLOAD_DIR):
        os.remove(os.path.join(UPLOAD_DIR, f))
    return {"message": "All PDFs cleared"}
