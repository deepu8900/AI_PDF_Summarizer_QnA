import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class RAGPipeline:

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.vector_store = None
        self.llm = self._load_local_llm()
        self.prompts = self._load_prompts()

    def _load_local_llm(self):
        model_path = "google/flan-t5-base"

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        hf_pipeline = pipeline(
            task="text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256
        )

        return HuggingFacePipeline(pipeline=hf_pipeline)

    def _load_prompts(self):
        prompt_path = os.path.join(
            os.path.dirname(__file__),
            "prompt_template.txt"
        )

        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()

        parts = content.split("QUESTION ANSWERING PROMPT")

        summary_prompt = parts[0]
        qa_prompt = "QUESTION ANSWERING PROMPT" + parts[1]

        return {
            "summary": summary_prompt,
            "qa": qa_prompt
        }

    def create_vector_store(self):
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

    def summarize(self):
        if not self.vector_store:
            raise ValueError("Vector store not created.")

        docs = self.vector_store.similarity_search("", k=5)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = PromptTemplate(
            template=self.prompts["summary"],
            input_variables=["context"]
        )

        final_prompt = prompt.format(context=context)

        response = self.llm.invoke(final_prompt)
        return response.strip()

    def answer_question(self, question: str):
        if not self.vector_store:
            raise ValueError("Vector store not created.")

        retriever = self.vector_store.as_retriever()

        prompt = PromptTemplate(
            template=self.prompts["qa"],
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )

        response = qa_chain.run(question)
        return response.strip()