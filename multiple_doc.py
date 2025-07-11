import os
import sqlite3
import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import tempfile
import google.generativeai as genai
from datetime import datetime
import re
from PIL import Image
from typing import List
from docx import Document
import html2text
import hashlib
import pandas as pd

# Constants
DB_PATH = "db/sections.db"
GEMINI_MODEL = "models/gemini-2.5-flash-preview-05-20"

# Initialize database
@st.cache_resource
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                header TEXT,
                content TEXT,
                page_number INTEGER,
                content_hash TEXT
            )
        ''')
    return conn

# Generate content hash
def content_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# Gemini prompt to extract sectioned content
SECTION_EXTRACTION_PROMPT = """
You are given the following page content from a document:

{text}

Extract all section headers and their content. For each section:
- Start the section with '### ' followed by the section header.
- Under each header, include all paragraph content exactly as in the document.
- If the content includes tables, equations, or structured data, preserve its format.
- Use **Markdown** formatting.
Return the result in structured Markdown format.
"""

# Extract structured content using Gemini
def extract_sections_with_gemini(text, page_number, filename, model):
    try:
        prompt = SECTION_EXTRACTION_PROMPT.format(text=text)
        response = model.generate_content(prompt)
        result = response.text

        sections = []
        lines = result.split('\n')
        current_header = None
        content_buffer = []

        for line in lines:
            if line.strip().startswith("### "):
                if current_header:
                    combined = "\n".join(content_buffer).strip()
                    hash_val = content_hash(combined)
                    sections.append({
                        "header": current_header.strip(),
                        "content": combined,
                        "page_number": page_number,
                        "filename": filename,
                        "content_hash": hash_val
                    })
                current_header = line.replace("### ", "").strip()
                content_buffer = []
            elif current_header:
                content_buffer.append(line)

        if current_header:
            combined = "\n".join(content_buffer).strip()
            hash_val = content_hash(combined)
            sections.append({
                "header": current_header.strip(),
                "content": combined,
                "page_number": page_number,
                "filename": filename,
                "content_hash": hash_val
            })

        return clean_and_deduplicate_sections(sections)
    except Exception as e:
        st.error(f"❌ Gemini Error on page {page_number}: {e}")
        return []

# Clean and deduplicate extracted sections
def clean_and_deduplicate_sections(sections):
    seen = set()
    cleaned = []
    for sec in sections:
        if sec['content_hash'] not in seen:
            sec['content'] = re.sub(r'\n{2,}', '\n', sec['content'])
            sec['content'] = re.sub(r'([A-Za-z])\n([A-Za-z])', r'\1\2', sec['content'])
            cleaned.append(sec)
            seen.add(sec['content_hash'])
    return cleaned

# Extract text from image-based PDF
def extract_text_from_image_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    return [pytesseract.image_to_string(img) for img in images]

# DOCX, TXT, HTML extractors
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return html2text.html2text(html_content)

# Generic processor for any file type
def process_single_file(file_path, filename, model, conn):
    cur = conn.cursor()
    ext = filename.split('.')[-1].lower()

    try:
        if ext == "pdf":
            try:
                reader = PdfReader(file_path)
                total_pages = len(reader.pages)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if not text.strip():
                        continue
                    sections = extract_sections_with_gemini(text, i + 1, filename, model)
                    for sec in sections:
                        cur.execute("""
                            INSERT OR IGNORE INTO sections (filename, header, content, page_number, content_hash)
                            VALUES (?, ?, ?, ?, ?)
                        """, (sec['filename'], sec['header'], sec['content'], sec['page_number'], sec['content_hash']))
                st.write(f"✅ {filename} Page {i + 1}/{total_pages} processed with {len(sections)} section(s).")
            except Exception:
                st.warning(f"Text not found in {filename}, attempting OCR...")
                pages = extract_text_from_image_pdf(file_path)
                for i, text in enumerate(pages):
                    if not text.strip():
                        continue
                    sections = extract_sections_with_gemini(text, i + 1, filename, model)
                    for sec in sections:
                        cur.execute("""
                            INSERT OR IGNORE INTO sections (filename, header, content, page_number, content_hash)
                            VALUES (?, ?, ?, ?, ?)
                        """, (sec['filename'], sec['header'], sec['content'], sec['page_number'], sec['content_hash']))
                st.write(f"✅ OCR {filename} processed.")

        elif ext == "docx":
            text = extract_text_from_docx(file_path)
            sections = extract_sections_with_gemini(text, 1, filename, model)
            for sec in sections:
                cur.execute("""
                    INSERT OR IGNORE INTO sections (filename, header, content, page_number, content_hash)
                    VALUES (?, ?, ?, ?, ?)
                """, (sec['filename'], sec['header'], sec['content'], sec['page_number'], sec['content_hash']))
        elif ext == "txt":
            text = extract_text_from_txt(file_path)
            sections = extract_sections_with_gemini(text, 1, filename, model)
            for sec in sections:
                cur.execute("""
                    INSERT OR IGNORE INTO sections (filename, header, content, page_number, content_hash)
                    VALUES (?, ?, ?, ?, ?)
                """, (sec['filename'], sec['header'], sec['content'], sec['page_number'], sec['content_hash']))
        elif ext == "html":
            text = extract_text_from_html(file_path)
            sections = extract_sections_with_gemini(text, 1, filename, model)
            for sec in sections:
                cur.execute("""
                    INSERT OR IGNORE INTO sections (filename, header, content, page_number, content_hash)
                    VALUES (?, ?, ?, ?, ?)
                """, (sec['filename'], sec['header'], sec['content'], sec['page_number'], sec['content_hash']))
        else:
            st.warning(f"❌ Unsupported file type: {ext}")
            return

        conn.commit()
        st.success(f"✅ {filename} processed.")
    except Exception as e:
        st.error(f"⚠️ Failed to process {filename}: {e}")

# Execute SQL safely
def run_sql_query(sql_query):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    try:
        if not sql_query.strip().lower().startswith("select"):
            return "Only SELECT queries are allowed.", []
        cur.execute(sql_query)
        results = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return columns, results
    except Exception as e:
        return f"Error: {e}", []
    finally:
        conn.close()

# Streamlit App UI
st.set_page_config(page_title="Multi-Format Document Section Extractor", layout="wide")
st.title("📑 Process Multiple Invoices (PDF, DOCX, TXT, HTML) with Gemini 2.5")

uploaded_files = st.file_uploader(
    "📁 Upload one or more invoice documents", 
    type=["pdf", "docx", "txt", "html"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("📤 Process All Documents"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyCAlwR4a6AHmXMgpao6tEp8Z5yPZbyQt7U"))
        model = genai.GenerativeModel(GEMINI_MODEL)
        conn = init_db()
        for file in uploaded_files:
            ext = file.name.split('.')[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(file.read())
                file_path = tmp.name
            filename = file.name
            process_single_file(file_path, filename, model, conn)
        # Do not close conn manually (avoid 'closed db' error)

# SQL Query Interface
st.header("🧠 SQL Query Interface")
sql_query = st.text_area("📜 Enter your SQL query (e.g., SELECT * FROM sections WHERE header LIKE '%Passenger Name%')")
if st.button("🔍 Run SQL Query"):
    columns, results = run_sql_query(sql_query)
    if isinstance(columns, str):
        st.error(columns)
    elif results:
        st.success(f"✅ {len(results)} row(s) found.")
        df = pd.DataFrame(results, columns=columns)
        st.dataframe(df)
    else:
        st.warning("No results found.")
