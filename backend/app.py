from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os

from rag_pipeline import RAGPipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_instance = None


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global pipeline_instance

    upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pipeline_instance = RAGPipeline(file_path)
    pipeline_instance.create_vector_store()

    summary = pipeline_instance.summarize()

    return JSONResponse(content={"summary": summary})


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    global pipeline_instance

    if pipeline_instance is None:
        return JSONResponse(content={"error": "Upload a PDF first."}, status_code=400)

    answer = pipeline_instance.answer_question(question)

    return JSONResponse(content={"answer": answer})