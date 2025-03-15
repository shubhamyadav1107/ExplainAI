import streamlit as st
import fitz  # PyMuPDF for PDF processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain_community.llms import Cohere
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import spacy
import graphviz

# Load environment variables
load_dotenv()

# Load spaCy model for key phrase extraction
nlp = spacy.load("en_core_web_sm")

# Streamlit UI
st.set_page_config(page_title="📝🤓ExplainAI👽: Transforming Learning With AI", layout="wide")
st.title("📝🤓ExplainAI👽: Transforming Learning With AI")
st.write("Drop your document, get an AI-generated summary, ask questions, and visualize key insights!")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function to extract key phrases
def extract_key_phrases(text):
    doc = nlp(text)
    key_phrases = [chunk.text for chunk in doc.noun_chunks]  # Extract noun phrases
    return key_phrases

# Function to generate flowchart using Graphviz
def generate_flowchart(key_phrases):
    dot = graphviz.Digraph()
    
    for i, phrase in enumerate(key_phrases):
        dot.node(str(i), phrase)  # Create node for each key phrase
    
    for i in range(len(key_phrases) - 1):
        dot.edge(str(i), str(i + 1))  # Connect nodes sequentially
    
    return dot

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file as temp.pdf
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Extract text from PDF
    extracted_text = extract_text_from_pdf(temp_path)

    # Extract key phrases
    key_phrases = extract_key_phrases(extracted_text)

    # Display key phrases
    st.subheader("📌 Key Phrases Extracted:")
    st.write(key_phrases)

    # Generate and display flowchart
    if st.button("Generate Flowchart"):
        flowchart = generate_flowchart(key_phrases)
        st.graphviz_chart(flowchart)

    # Split text into chunks for retrieval-based QA
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
