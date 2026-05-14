import os
import json
import time
import fitz
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def read_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=3000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarise_chunk(chunk, chunk_number, total_chunks):
    prompt = f"""You are an expert teacher helping a student prepare for their upcoming exam.
You are summarising part {chunk_number} of {total_chunks} of their study material.

Here is the content:
{chunk}

From this section extract and present:
1. The main topic being covered
2. Key concepts and definitions the student must know — explained in simple clear language
3. Important points likely to appear in an exam — marked with "EXAM IMPORTANT:"
4. If there are any answers in the text, highlight the key answer points clearly
5. A one line memory tip or trick if possible

Write as if you are a friendly teacher explaining to a stressed student the night before their exam.
Keep it clear, simple and focused."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def create_final_summary(chunk_summaries):
    
    batch_size = 8
    batches = [chunk_summaries[i:i+batch_size] for i in range(0, len(chunk_summaries), batch_size)]
    
    batch_summaries = []
    for i, batch in enumerate(batches):
        combined = "\n\n".join(batch)
        prompt = f"""Summarise these study notes into 5 key points only.
Be very concise — maximum 3 sentences per point.

{combined}"""
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        batch_summaries.append(response.choices[0].message.content)
        time.sleep(15)

    all_summaries = "\n\n".join(batch_summaries)

    prompt = f"""You are an expert teacher creating a final exam preparation guide.

Here are condensed summaries of all sections:
{all_summaries}

Create the ultimate exam preparation guide with:

1. WHAT THIS SUBJECT IS ABOUT
   Simple paragraph in plain language

2. MOST IMPORTANT TOPICS TO STUDY
   Top 8 to 10 topics most likely in the exam

3. KEY DEFINITIONS TO MEMORISE
   All important terms with simple definitions

4. LIKELY EXAM QUESTIONS
   8 to 10 questions that could appear in the exam

5. QUICK REVISION POINTS
   15 to 20 rapid fire points to read 30 minutes before exam

6. TOPICS TO NOT SKIP
   Topics that appeared multiple times

Write in a friendly encouraging tone."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

st.set_page_config(page_title="Exam Prep AI", page_icon="📚", layout="centered")

st.title("📚 Exam Preparation Guide Generator")
st.write("Upload your study notes and get a complete exam preparation guide instantly.")

uploaded_file = st.file_uploader("Upload your PDF notes", type=["pdf"])

if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")

    if st.button("Generate Exam Guide"):
        with st.spinner("Reading your PDF..."):
            text = read_pdf(uploaded_file)
            st.info(f"Read {len(text)} characters from your document")

        chunks = chunk_text(text)
        st.info(f"Split into {len(chunks)} sections for processing")

        chunk_summaries = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, chunk in enumerate(chunks):
            status_text.text(f"Analysing section {i+1} of {len(chunks)}...")
            summary = summarise_chunk(chunk, i+1, len(chunks))
            chunk_summaries.append(summary)
            progress_bar.progress((i+1) / len(chunks))
            time.sleep(15)

        status_text.text("Creating your exam guide...")

        with st.spinner("Putting it all together..."):
            final_guide = create_final_summary(chunk_summaries)

        st.success("Your exam guide is ready!")
        st.markdown("---")
        st.markdown(final_guide)
        st.markdown("---")

        st.download_button(
            label="Download Exam Guide",
            data=final_guide,
            file_name="exam_guide.txt",
            mime="text/plain"
        )