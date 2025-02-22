import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

# Load pre-trained model for question-answering
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF"""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    return text

# Streamlit UI
st.title("PaperBot.ai")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    question = st.text_input("Ask a question:")
    
    if question:
        answer = qa_pipeline(question=question, context=pdf_text)
        st.write("### Answer:", answer["answer"])
