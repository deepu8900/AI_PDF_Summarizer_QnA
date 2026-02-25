from transformers import pipeline
from langchain.llms import HuggingFacePipeline

def get_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512
    )

    return HuggingFacePipeline(pipeline=pipe)