import os
import re
import streamlit as st
from transformers import pipeline
from diffusers import DiffusionPipeline
from PIL import Image
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation

# Title of the app
st.title(" ExplainAI: Transforming ")

# Sidebar instructions
st.sidebar.header("Upload Files")
st.sidebar.write("Upload a PDF, Word, or PPT file to generate visual insights.")

# File upload section
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "pptx"])
if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() if page.extract_text() else ""
    return text

# Function to extract text from Word
def extract_text_from_word(file):
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Function to extract text from PPT
def extract_text_from_ppt(file):
    prs = Presentation(file)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

# Function to summarize text
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text, max_length=500):
    summarizer = load_summarizer()
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

# Function to break down text into key phrases
def break_down_text(text):
    sentences = re.split(r"[.!?]", text)
    key_phrases = [sentence.strip() for sentence in sentences if sentence.strip()]
    return key_phrases

# Stable Diffusion model loading
@st.cache_resource
def load_diffusion_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype="auto",
    )
    return pipe.to("cpu")  # Use CPU for compatibility with Streamlit

# Function to generate image from text
def generate_image_from_text(prompt, pipeline):
    result = pipeline(prompt).images[0]
    filename = f"{prompt[:10].replace(' ', '_')}.png"
    result.save(filename)
    return filename

# Processing the uploaded file
if uploaded_file is not None:
    st.write(f"Processing {uploaded_file.name}...")

    # Extract text based on file type
    if file_extension == "pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        extracted_text = extract_text_from_word(uploaded_file)
    elif file_extension == "pptx":
        extracted_text = extract_text_from_ppt(uploaded_file)
    else:
        st.error("Unsupported file format!")
        st.stop()

    # Display extracted text
    st.subheader("Extracted Text")
    st.write(extracted_text)

    # Summarize the text
    summarized_text = summarize_text(extracted_text)
    st.subheader("Summarized Text")
    st.write(summarized_text)

    # Generate key phrases
    key_phrases = break_down_text(summarized_text)
    st.subheader("Key Phrases")
    st.write(key_phrases)

    # Load Stable Diffusion pipeline
    diffusion_pipeline = load_diffusion_pipeline()

    # Generate images for each key phrase
    st.subheader("Generated Images")
    for phrase in key_phrases:
        try:
            st.write(f"Generating image for: '{phrase}'")
            image_path = generate_image_from_text(phrase, diffusion_pipeline)
            st.image(Image.open(image_path), caption=phrase, use_column_width=True)
        except Exception as e:
            st.error(f"Error generating image for '{phrase}': {e}")
