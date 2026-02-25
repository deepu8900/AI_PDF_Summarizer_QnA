from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.rag_pipeline import (
    save_pdf,
    create_vectorstore,
    summarize_document,
    ask_question
)

app = FastAPI(title="AI PDF RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "RAG API running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    path = save_pdf(content, file.filename)
    create_vectorstore(path)
    return {"message": "PDF processed successfully"}


@app.get("/summarize")
async def summarize():
    result = summarize_document()
    return {"summary": result}


@app.post("/ask")
async def ask(question: str):
    answer = ask_question(question)
    return {"answer": answer}