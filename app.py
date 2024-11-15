import os
import re
from io import BytesIO
from PIL import Image
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from transformers import pipeline
from pptx import Presentation
from docx import Document
import PyPDF2


# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() if page.extract_text() else ""
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""


# Function to extract text from Word file
def extract_text_from_word(file_path):
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from Word file: {e}")
        return ""


# Function to extract text from PPT file
def extract_text_from_ppt(file_path):
    try:
        prs = Presentation(file_path)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PPT file: {e}")
        return ""


# Function to summarize text using Hugging Face pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


def summarize_text(text, max_length=500):
    summarizer = load_summarizer()
    try:
        summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return ""


# Function to split text into key phrases
def break_down_text(text):
    sentences = re.split(r"[.!?]", text)
    key_phrases = [sentence.strip() for sentence in sentences if sentence.strip()]
    return key_phrases


# Load Stable Diffusion model
@st.cache_resource
def load_stable_diffusion():
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe = pipe.to(device)
        return pipe
    except Exception as e:
        st.error(f"Error loading Stable Diffusion model: {e}")
        return None


# Function to generate images from text
def generate_image_from_text(pipe, prompt):
    if not prompt.strip():
        return None, "Prompt is empty or invalid."
    try:
        image = pipe(prompt).images[0]
        return image, None
    except Exception as e:
        return None, f"Error generating image for '{prompt}': {e}"


# Streamlit App
st.title(" 📝🤓ExplainAI👽: Transforming Learning with AI ")
st.write("Upload a file and visualize its content as summarized text and images.")

uploaded_file = st.file_uploader("Upload a PDF, Word, or PPT file", type=["pdf", "docx", "pptx"])

if uploaded_file:
    # Extract text based on file type
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        text = extract_text_from_word(uploaded_file)
    elif file_extension == "pptx":
        text = extract_text_from_ppt(uploaded_file)
    else:
        st.error("Unsupported file type.")
        text = ""

    if text:
        st.subheader("Extracted Text")
        st.write(text)

        # Summarize text
        summarized_text = summarize_text(text)
        if summarized_text:
            st.subheader("Summarized Text")
            st.write(summarized_text)

            # Extract key phrases
            key_phrases = break_down_text(summarized_text)
            if key_phrases:
                st.subheader("Key Phrases")
                st.write(key_phrases)

                # Generate images
                st.subheader("Generated Images")
                stable_diffusion_pipe = load_stable_diffusion()
                if stable_diffusion_pipe:
                    for phrase in key_phrases:
                        image, error = generate_image_from_text(stable_diffusion_pipe, phrase)
                        if error:
                            st.error(error)
                        else:
                            st.image(image, caption=phrase)
            else:
                st.warning("No key phrases extracted.")
        else:
            st.warning("Text summarization failed.")
    else:
        st.warning("No text extracted from the uploaded file.")
