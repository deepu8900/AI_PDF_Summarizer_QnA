"""
RAG Pipeline using:
- LangChain for document loading & chaining
- FAISS for vector storage
- sentence-transformers for embeddings (free, local)
- Mistral-7B-Instruct via HuggingFace Inference API (free tier)
"""

import os
from typing import List, Optional, Tuple, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from backend.prompts import SUMMARIZE_PROMPT, QNA_PROMPT, CONDENSE_QUESTION_PROMPT


class RAGPipeline:
    def __init__(self):
        self.hf_token = os.environ.get("HF_TOKEN", "")

        
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

       
        print("Connecting to Mistral-7B...")
        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=self.hf_token,
            max_new_tokens=1024,
            temperature=0.3,
            repetition_penalty=1.1,
        )

        self.vectorstore: Optional[FAISS] = None
        self.loaded_pdfs: Dict[str, dict] = {}  
        self.all_docs: List[Document] = []

       
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
            separators=["\n\n", "\n", ".", "!", "?", " "]
        )

    def ingest_pdf(self, file_path: str, pdf_id: str, filename: str) -> int:
        """Load and index a PDF. Returns chunk count."""
        loader = PyPDFLoader(file_path)
        pages = loader.load()

       
        for doc in pages:
            doc.metadata["pdf_id"] = pdf_id
            doc.metadata["filename"] = filename

        chunks = self.splitter.split_documents(pages)

        self.all_docs.extend(chunks)
        self.loaded_pdfs[pdf_id] = {"filename": filename, "chunk_count": len(chunks)}

        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        print(f"Ingested '{filename}' → {len(chunks)} chunks")
        return len(chunks)

    def summarize(self, pdf_ids: Optional[List[str]] = None) -> str:
        """Generate a summary. If pdf_ids given, only summarize those PDFs."""
        if pdf_ids:
            docs = [d for d in self.all_docs if d.metadata.get("pdf_id") in pdf_ids]
        else:
            docs = self.all_docs

        if not docs:
            return "No documents found to summarize."

        
        combined_text = "\n\n".join([d.page_content for d in docs[:20]])[:3000]

        filenames = list({d.metadata.get("filename", "unknown") for d in docs})
        file_list = ", ".join(filenames)

        prompt = SUMMARIZE_PROMPT.format(
            filenames=file_list,
            content=combined_text
        )

        response = self.llm.invoke(prompt)
        return response.strip()

    def answer_question(
        self,
        question: str,
        chat_history: Optional[List[dict]] = None
    ) -> Tuple[str, List[dict]]:
        """Answer a question using RAG. Returns (answer, sources)."""
        if self.vectorstore is None:
            return "No documents loaded.", []

       
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        relevant_docs = retriever.invoke(question)
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

        
        history_str = ""
        if chat_history:
            for msg in chat_history[-6:]:  
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    history_str += f"Human: {content}\n"
                elif role == "assistant":
                    history_str += f"Assistant: {content}\n"

        prompt = QNA_PROMPT.format(
            chat_history=history_str,
            context=context,
            question=question
        )

        answer = self.llm.invoke(prompt)
        answer = answer.strip()

        # Build source list
        sources = []
        seen = set()
        for doc in relevant_docs:
            key = (doc.metadata.get("filename", ""), doc.metadata.get("page", 0))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "filename": doc.metadata.get("filename", "unknown"),
                    "page": doc.metadata.get("page", 0) + 1,
                    "snippet": doc.page_content[:120] + "..."
                })

        return answer, sources

    def get_loaded_pdfs(self) -> Dict[str, dict]:
        return self.loaded_pdfs

    def remove_pdf(self, pdf_id: str) -> bool:
        if pdf_id not in self.loaded_pdfs:
            return False

        # Remove from tracking
        del self.loaded_pdfs[pdf_id]
        self.all_docs = [d for d in self.all_docs if d.metadata.get("pdf_id") != pdf_id]

        # Rebuild vector store from remaining docs
        if self.all_docs:
            self.vectorstore = FAISS.from_documents(self.all_docs, self.embeddings)
        else:
            self.vectorstore = None

        return True

    def reset(self):
        """Clear everything."""
        self.vectorstore = None
        self.loaded_pdfs = {}
        self.all_docs = []
