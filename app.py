import os
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image

# Helper functions for text extraction
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text

def extract_text_from_word(file_path):
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_ppt(file_path):
    prs = Presentation(file_path)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

# Summarization function
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text, max_length=500):
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

# Text breakdown into key phrases
def break_down_text(text):
    import re
    sentences = re.split(r'[.!?]', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Stable Diffusion model setup
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")

def generate_image_from_text(prompt):
    image = pipe(prompt).images[0]
    filename = f"{prompt[:10]}.png"
    image.save(filename)
    return filename

# Streamlit app
st.title("📝🤓ExplainAI👽")
st.write("Upload a file and visualize its content as summarized text and images.")

uploaded_file = st.file_uploader("Upload your file (PDF, Word, or PPT)", type=["pdf", "docx", "pptx"])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1]
    if file_type == "pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "docx":
        extracted_text = extract_text_from_word(uploaded_file)
    elif file_type == "pptx":
        extracted_text = extract_text_from_ppt(uploaded_file)
    else:
        st.error("Unsupported file type.")

    st.subheader("Extracted Text")
    st.write(extracted_text)

    summarized_text = summarize_text(extracted_text)
    st.subheader("Summarized Text")
    st.write(summarized_text)

    key_phrases = break_down_text(summarized_text)
    st.subheader("Key Phrases")
    st.write(key_phrases)

    st.subheader("Generated Images")
    for phrase in key_phrases:
        try:
            filename = generate_image_from_text(phrase)
            st.image(filename, caption=phrase)
        except Exception as e:
            st.error(f"Error generating image for '{phrase}': {e}")
