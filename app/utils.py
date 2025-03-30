from pypdf import PdfReader
from docx import Document
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import time

def load_sds(file_path: str):
    start_time = time.time()
    text = ""

    if file_path.lower().endswith('.pdf'):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""

    elif file_path.lower().endswith('.docx'):
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"

        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                text += row_text + "\n"

    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    end_time = time.time()
    print(f"Document Processing Time: {end_time - start_time:.2f} sec")
    return text

def preprocess_text(text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    if not chunks and text.strip():
        chunks = [text.strip()]
    
    return chunks
