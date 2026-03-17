"""
prompts.py — All prompt templates for the PDF Summarizer & QnA system.

Design principles:
- Clear role framing with [INST] tags for Mistral
- Explicit output format instructions
- Grounded answers only (no hallucination)
- Concise, useful outputs
"""


SUMMARIZE_PROMPT = """<s>[INST]
You are an expert document analyst. Your job is to produce a clear, structured summary of the provided document(s).

Document(s): {filenames}

Content:
{content}

Instructions:
- Write a concise executive summary (2-3 sentences)
- List the 4-6 key points or main topics covered
- Note any important conclusions, findings, or recommendations
- Keep language clear and professional
- Do NOT add information that is not in the document

Format your response as:

## Summary
[2-3 sentence overview]

## Key Points
- [point 1]
- [point 2]
- [point 3]
...

## Conclusions / Findings
[Any conclusions or notable outcomes from the document]
[/INST]"""



QNA_PROMPT = """<s>[INST]
You are a helpful AI assistant that answers questions based strictly on the provided document context.

Previous conversation:
{chat_history}

Relevant document excerpts:
{context}

User's question: {question}

Instructions:
- Answer ONLY using information from the document excerpts above
- If the answer is not found in the excerpts, say: "I couldn't find information about this in the uploaded documents."
- Be direct and concise — get to the point
- If helpful, reference which part of the document supports your answer
- For follow-up questions, use the conversation history for context
- Never make up facts or go beyond what the documents say
[/INST]"""



CONDENSE_QUESTION_PROMPT = """<s>[INST]
Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that captures all relevant context.

Chat History:
{chat_history}

Follow-up question: {question}

Standalone question (output only the rephrased question, nothing else):
[/INST]"""



SYSTEM_NOTES = """
Model: mistralai/Mistral-7B-Instruct-v0.2
Prompt format: <s>[INST] ... [/INST]
Temperature: 0.3  (low = more factual, less creative)
Max tokens: 1024

Tips for improving responses:
- Increase chunk_size in rag.py if context feels too fragmented
- Lower temperature (0.1) for more deterministic summarization
- Add few-shot examples inside the prompt for specific output styles
- Use pdf_ids filter in /summarize to target specific documents
"""
