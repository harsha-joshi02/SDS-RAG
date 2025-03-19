from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from typing import List, Optional
from pathlib import Path
import os
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
    print(f"Received sds_paths: {sds_paths}")
    if not sds_paths:
        sds_paths = [str(f) for f in UPLOAD_DIR.glob("*.pdf")]
        if not sds_paths:
            raise HTTPException(status_code=400, detail="No SDS files available. Upload files or submit URLs first.")
    for path in sds_paths:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
    rag_system = RAGSystem(sds_paths)
    answer = rag_system.query(question)
    return {"question": question, "answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)