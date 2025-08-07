from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text_chunks(chunks):
    return model.encode(chunks, convert_to_tensor=True)

def get_answer_from_question(question, chunks, embeddings):
    question_embedding = model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, embeddings)[0]
    top_idx = similarities.argmax().item()
    return chunks[top_idx]
