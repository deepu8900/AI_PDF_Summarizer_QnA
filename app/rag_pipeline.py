import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from app.embeddings import get_embeddings
from app.llm import get_llm
from app.prompts import summary_prompt, qa_prompt

UPLOAD_DIR = "uploads"
VECTOR_DIR = "vectorstore"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)


def save_pdf(file_bytes: bytes, filename: str):
    file_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{file_id}_{filename}")
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path


def create_vectorstore(path: str):
    loader = PyPDFLoader(path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DIR)

    return vectorstore


def load_vectorstore():
    embeddings = get_embeddings()
    return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)


def summarize_document():
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search("", k=5)

    context = "\n".join([doc.page_content for doc in docs])

    llm = get_llm()

    prompt = summary_prompt.format(context=context)
    return llm.invoke(prompt)


def ask_question(question: str):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(question, k=4)

    context = "\n".join([doc.page_content for doc in docs])

    llm = get_llm()

    prompt = qa_prompt.format(context=context, question=question)
    return llm.invoke(prompt)