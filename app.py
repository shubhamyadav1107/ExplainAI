import streamlit as st
import fitz  # PyMuPDF for PDF processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_community.llms import Cohere
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="📝🤓ExplainAI👽: Transforming Learning With AI", layout="wide")
st.title("📝🤓ExplainAI👽: Transforming Learning With AI")
st.write("Drop your document, get an AI-generated summary, ask questions, and visualize key insights!")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file as temp.pdf
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Extract text from PDF
    def extract_text_from_pdf(pdf_path):
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
        return text

    extracted_text = extract_text_from_pdf(temp_path)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(extracted_text)

    # Ensure API key is available
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        st.error("⚠️ Cohere API key is missing! Please add it to your environment variables in Streamlit Cloud.")
    else:
        # Initialize FAISS and Cohere Embeddings
        embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=cohere_api_key)
        vector_store = FAISS.from_texts(text_chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # Initialize Cohere model
        llm = Cohere(model="command", cohere_api_key=cohere_api_key, temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # User input for questions
        user_query = st.text_input("Ask a question about the document:")

        if user_query:
            response = qa_chain.run(user_query)
            st.write("### 🤖 Answer:")
            st.write(response)
