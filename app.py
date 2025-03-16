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
import re

# Load environment variables
load_dotenv()

# Download and load spaCy model
os.system("python -m spacy download en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Initialize session state
if "extracted_text" not in st.session_state:
    st.session_state["extracted_text"] = ""
if "key_phrases" not in st.session_state:
    st.session_state["key_phrases"] = []
if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = None

# Streamlit UI
st.set_page_config(page_title="📝🤓ExplainAI👽: Transforming Learning With AI", layout="wide")
st.title("📝🤓ExplainAI👽: Transforming Learning With AI")
st.write("Drop your document, get an AI-generated summary, ask questions, and visualize key insights!")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"❌ Error extracting text: {e}")
        return None

# Improved function to extract meaningful key phrases
def extract_key_phrases(text):
    doc = nlp(text)
    key_phrases = set()

    # Extract Named Entities (important concepts)
    for ent in doc.ents:
        key_phrases.add(ent.text)

    # Extract Noun Chunks (technical terms, important phrases)
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if len(phrase.split()) > 2:  # Avoid very short phrases
            key_phrases.add(phrase)

    # Extract Verb Phrases (actionable insights)
    for token in doc:
        if token.pos_ == "VERB" and token.text.lower() not in ["is", "are", "was", "were", "be"]:
            key_phrases.add(token.lemma_)  # Use base form of the verb

    # Clean key phrases
    cleaned_phrases = set()
    for phrase in key_phrases:
        phrase = re.sub(r'\s+', ' ', phrase).strip()  # Remove extra spaces
        if phrase and len(phrase) > 3:  # Ignore very short phrases
            cleaned_phrases.add(phrase)

    return list(cleaned_phrases)

# Function to generate flowchart with better structure
def generate_flowchart(key_phrases):
    dot = graphviz.Digraph()
    
    if not key_phrases:
        return dot

    # Create nodes
    for i, phrase in enumerate(key_phrases):
        dot.node(str(i), phrase)

    # Link related phrases
    for i in range(len(key_phrases) - 1):
        dot.edge(str(i), str(i + 1))  # Basic sequential connection

    return dot

# File uploader
uploaded_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    
    # Step 1: Extract text from PDF
    if st.button("📄 Extract Text"):
        temp_path = "temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        extracted_text = extract_text_from_pdf(temp_path)
        
        if extracted_text:
            st.session_state["extracted_text"] = extracted_text  # Store extracted text in session
            st.text_area("📃 Extracted Text:", extracted_text, height=300)
        else:
            st.error("❌ Failed to extract text. Please try another document.")
    
    # Step 2: Extract Key Phrases
    if st.session_state["extracted_text"] and st.button("🔍 Extract Key Phrases"):
        st.session_state["key_phrases"] = extract_key_phrases(st.session_state["extracted_text"])
        
        if st.session_state["key_phrases"]:
            st.subheader("📌 Key Phrases Extracted:")
            st.write(st.session_state["key_phrases"])
        else:
            st.error("❌ No key phrases found.")

    # Step 3: Generate Flowchart
    if st.session_state["key_phrases"] and st.button("📊 Generate Flowchart"):
        flowchart = generate_flowchart(st.session_state["key_phrases"])
        st.graphviz_chart(flowchart)

    # Step 4: Setup QA Model
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        st.error("⚠️ Cohere API key is missing! Please add it to your environment variables.")
    else:
        if st.session_state["extracted_text"] and st.button("🤖 Setup AI Q&A"):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            text_chunks = text_splitter.split_text(st.session_state["extracted_text"])

            embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=cohere_api_key)
            vector_store = FAISS.from_texts(text_chunks, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})

            llm = Cohere(model="command", cohere_api_key=cohere_api_key, temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            st.session_state["qa_chain"] = qa_chain  # Store QA model in session state
            st.success("✅ AI Q&A model is ready!")

    # Step 5: Ask Questions
    if st.session_state["qa_chain"]:
        user_query = st.text_input("📝 Ask a question about the document:")
        if user_query:
            response = st.session_state["qa_chain"].run(user_query)
            st.write("### 🤖 Answer:")
            st.write(response)
