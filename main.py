from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from typing import List, Optional
from pathlib import Path
import os
import time
import uvicorn
from app.rag import RAGSystem
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_system.log")
    ]
)
logger = logging.getLogger(__name__)

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
        logger.info(f"Uploaded file: {file.filename}")
    return {"file_paths": file_paths}

@app.post("/submit-url/")
async def submit_url(url: str = Query(...)):
    logger.info(f"URL submission request: {url}")
    from app.search import search_url
    result = search_url(url)
    return result

@app.post("/query/")
async def query_rag(question: str = Query(...), sds_paths: Optional[List[str]] = Query(None)):
    logger.info(f"Query request: {question}")
    start_time = time.time()

    if not sds_paths:
        sds_paths = [str(f) for f in UPLOAD_DIR.glob("*.pdf")] + [str(f) for f in UPLOAD_DIR.glob("*.docx")]
        logger.info(f"Found {len(sds_paths)} documents in upload directory")

    if not sds_paths:
        try:
            rag_system = RAGSystem([])
            if not rag_system.vectorstore or len(rag_system.vectorstore.docstore._dict) == 0:
                logger.warning("No data available for querying")
                raise HTTPException(status_code=400, detail="No data available. Upload files or submit URLs first.")
        except Exception as e:
            logger.error(f"Error accessing FAISS: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error accessing FAISS: {str(e)}")

    rag_system = RAGSystem(sds_paths)
    answer = rag_system.query(question)
    end_time = time.time()
    logger.info(f"API Response Time: {end_time - start_time:.2f} sec")

    return {"question": question, "answer": answer}

if __name__ == "__main__":
    logger.info("Starting RAG System server")
    uvicorn.run(app, host="0.0.0.0", port=8000)