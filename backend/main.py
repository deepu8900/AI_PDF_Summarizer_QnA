from fastapi import FastAPI, UploadFile, File, Form
from pdf_utils import extract_text_from_pdf, split_text_into_chunks
from qna_utils import embed_text_chunks, get_answer_from_question

app = FastAPI()

EMBEDDINGS = None
CHUNKS = None

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global EMBEDDINGS, CHUNKS
    contents = await file.read()
    with open("uploaded.pdf", "wb") as f:
        f.write(contents)

    text = extract_text_from_pdf("uploaded.pdf")
    CHUNKS = split_text_into_chunks(text)
    EMBEDDINGS = embed_text_chunks(CHUNKS)
    return {"message": "PDF uploaded and processed"}

@app.post("/ask-question/")
async def ask_question(question: str = Form(...)):
    if EMBEDDINGS is None or CHUNKS is None:
        return {"answer": "Please upload a PDF first."}
    answer = get_answer_from_question(question, CHUNKS, EMBEDDINGS)
    return {"answer": answer}
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# .\venv\Scripts\activate
# You’ll see:

# scss
# Copy
# Edit
# (venv) PS C:\Users\DELL\AI_PDF_Summarizer_QnA>
# 🔹 Step 4: Move to frontend folder
# powershell
# Copy
# Edit
# cd frontend
# 🔹 Step 5: Run the Streamlit app
# powershell
# Copy
# Edit
# streamlit run app.py
# Your web app will now open at:

# arduino
# Copy
# Edit
# http://localhost:8501
# Click the link or open it in your browser.