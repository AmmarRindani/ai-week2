import os
import json
import fitz
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def read_pdf(file_path):
    doc = fitz.open(file_path)
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
    prompt = f"""You are a expert teacher helping a student prepare for their upcoming exam.
You are summarising part {chunk_number} of {total_chunks} of their study material.

Here is the content:
{chunk}

From this section extract and present:
1. The main topic being covered
2. Key concepts and definitions the student must know — explained in simple clear language
3. Important points likely to appear in an exam — marked with "EXAM IMPORTANT:"
4. If there are any answers in the text, highlight the key answer points clearly
5. A one line memory tip or trick if possible to help remember something important

Write as if you are a friendly teacher explaining to a stressed student the night before their exam.
Keep it clear, simple and focused. No unnecessary information."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def create_final_summary(chunk_summaries):
    all_summaries = "\n\n".join(chunk_summaries)

    prompt = f"""You are an expert teacher creating a final exam preparation guide for a student.

You have summaries of all sections of their study material:
{all_summaries}

Now create the ultimate exam preparation guide with these sections:

1. WHAT THIS SUBJECT IS ABOUT
   Write a simple paragraph explaining the subject in plain language

2. MOST IMPORTANT TOPICS TO STUDY
   List the top 8 to 10 topics that are most likely to appear in the exam

3. KEY DEFINITIONS TO MEMORISE
   List all important terms and their simple definitions

4. LIKELY EXAM QUESTIONS
   Write 8 to 10 questions that could appear in the exam based on this material

5. QUICK REVISION POINTS
   A rapid fire list of 15 to 20 short points the student can read 30 minutes before the exam

6. TOPICS TO NOT SKIP
   List any topics that appeared multiple times or seem very important

Write in a friendly encouraging tone.
The student is stressed and needs clear focused help.
Make this guide something they can study from directly."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def save_to_file(content, filename="exam_guide.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved to {filename}")

chunk_summaries_file = "chunk_summaries.json"

if os.path.exists(chunk_summaries_file):
    print("Found saved chunk summaries — skipping PDF processing\n")
    with open(chunk_summaries_file, "r", encoding="utf-8") as f:
        chunk_summaries = json.load(f)
    print(f"Loaded {len(chunk_summaries)} chunk summaries\n")
else:
    print("Reading PDF...\n")
    text = read_pdf("notes.pdf")
    print(f"Total characters read: {len(text)}\n")

    print("Splitting into chunks...\n")
    chunks = chunk_text(text)
    print(f"Total chunks: {len(chunks)}\n")

    import time
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarising chunk {i+1} of {len(chunks)}...")
        summary = summarise_chunk(chunk, i+1, len(chunks))
        chunk_summaries.append(summary)
        with open(chunk_summaries_file, "w", encoding="utf-8") as f:
            json.dump(chunk_summaries, f)
        time.sleep(15)

print("Creating your exam guide...\n")
final_summary = create_final_summary(chunk_summaries)

print("\n=== EXAM PREPARATION GUIDE ===\n")
print(final_summary)

save_to_file(final_summary)