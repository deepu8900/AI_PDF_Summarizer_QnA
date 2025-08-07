import streamlit as st
import requests

st.title("📄 AI PDF Summarizer & QnA")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    with st.spinner("Uploading and processing PDF..."):
        response = requests.post(
            "http://localhost:8000/upload-pdf/",
            files={"file": open("temp.pdf", "rb")}
        )
    st.success("PDF processed. You can now ask questions!")

question = st.text_input("Ask a question from the PDF:")

if st.button("Get Answer") and question:
    with st.spinner("Thinking..."):
        response = requests.post(
            "http://localhost:8000/ask-question/",
            data={"question": question}
        )
    if response.status_code == 200:
        st.write("💡 Answer:", response.json()["answer"])
    else:
        st.error("Error: Could not fetch the answer.")
