from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from typing import List, Optional
from pathlib import Path
import os
import time
import uvicorn
from app.rag import RAGSystem

app = FastAPI()

UPLOAD_DIR = Path("data")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-sds/")
async def upload_sds(files: List[UploadFile] = File(...)):
    file_paths = []
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())
        file_paths.append(str(file_path))
    return {"file_paths": file_paths}

@app.post("/submit-url/")
async def submit_url(url: str = Query(...)):
    from app.search import search_url
    search_url(url)
    return {"url": url}

@app.post("/query/")
async def query_rag(question: str = Query(...), sds_paths: Optional[List[str]] = Query(None)):
    start_time = time.time()

    if not sds_paths:
        sds_paths = [str(f) for f in UPLOAD_DIR.glob("*.pdf")] + [str(f) for f in UPLOAD_DIR.glob("*.docx")]

    if not sds_paths:
        try:
            rag_system = RAGSystem([])
            if not rag_system.vectorstore or len(rag_system.vectorstore.docstore._dict) == 0:
                raise HTTPException(status_code=400, detail="No data available. Upload files or submit URLs first.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error accessing FAISS: {str(e)}")

    rag_system = RAGSystem(sds_paths)
    answer = rag_system.query(question)
    end_time = time.time()
    print(f"API Response Time: {end_time - start_time:.2f} sec")

    return {"question": question, "answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
