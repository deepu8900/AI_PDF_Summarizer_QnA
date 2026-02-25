from langchain.prompts import PromptTemplate

summary_prompt = PromptTemplate(
    template="""
You are an expert AI assistant.

Summarize the following document clearly and concisely.

Document:
{context}

Summary:
""",
    input_variables=["context"],
)

qa_prompt = PromptTemplate(
    template="""
You are an intelligent assistant.

Answer the question using ONLY the context below.
If the answer is not in the context, say "Answer not found in document."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)