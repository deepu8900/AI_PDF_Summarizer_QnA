AI PDF Summarizer using RAG
Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline to:

Upload PDF documents

Convert them into embeddings

Store them in a FAISS vector database

Answer user questions using a local LLM (FLAN-T5)

Architecture

User → FastAPI → PDF Loader → Text Splitter → Embeddings → FAISS → LLM → Response

Tech Stack

FastAPI

LangChain

FAISS

HuggingFace Transformers

FLAN-T5

Sentence Transformers

How It Works

Upload PDF

Text is chunked

Embeddings created

Stored in FAISS vector DB

Question is matched with relevant chunks

LLM generates contextual answer

Future Improvements

Frontend UI

Docker support

Deployment on AWS

Streaming responses

Authentication