from pypdf import PdfReader
from docx import Document
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import time
import pdfplumber

def load_sds(file_path: str):
    start_time = time.time()
    text = ""

    if file_path.lower().endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"

                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        text += " | ".join(str(cell).strip() if cell else "" for cell in row) + "\n"

    elif file_path.lower().endswith('.docx'):
        doc = Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():  # Avoid adding empty lines
                text += para.text.strip() + "\n"

        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() if cell.text.strip() else " " for cell in row.cells)
                text += row_text + "\n"

    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    end_time = time.time()
    print(f"Document Processing Time: {end_time - start_time:.2f} sec")
    return text

def preprocess_text(text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100,
        # separators=["\n\n", "\n", ".", "!", "?", ",", " ", "|"] 
    )
    chunks = text_splitter.split_text(text)

    if not chunks and text.strip():
        chunks = [text.strip()]
    
    return chunks   