from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from typing import List, Optional
from pathlib import Path
import os
import time
import uvicorn
import logging
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from pydantic import BaseModel

from app.config import CONFIG
from app.search import search_url
from app.graph import run_agent_workflow 
from app.excel_processor import ExcelToSQLProcessor
from app.web_search import WebSearchAgent
from app.cache import get_cached_response, set_cached_response  # Import updated cache functions

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_system.log")
    ]
)
logger = logging.getLogger(__name__)

excel_processor = ExcelToSQLProcessor(db_path="excel_data.sqlite")
schema_map = {}  # Maps schema names to list of table names

UPLOAD_DIR = Path(CONFIG["app"]["upload_dir"])
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic model for /set-schema/ request body
class SchemaRequest(BaseModel):
    schema_name: str
    tables: List[str]
    file_name: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    excel_processor.close()
    logger.info("Excel processor connection closed")

app = FastAPI(title=CONFIG["app"]["name"], lifespan=lifespan)

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
    result = search_url(url)
    return result

@app.post("/upload-excel/")
async def upload_excel(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx or .xls)")

    file_path = UPLOAD_DIR / file.filename

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        tables = excel_processor.process_excel_file(str(file_path))

        return {
            "file": file.filename,
            "tables_created": tables,
            "message": f"Excel file processed successfully. Created {len(tables)} tables."
        }
    except Exception as e:
        logger.error(f"Error processing Excel file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing Excel file: {str(e)}")

@app.post("/set-schema/")
async def set_schema(request: SchemaRequest):
    logger.info(f"Received set-schema request for schema_name: {request.schema_name}, tables: {request.tables}, file: {request.file_name}")
    try:
        if request.schema_name in schema_map:
            logger.warning(f"Schema name '{request.schema_name}' already exists")
            raise HTTPException(status_code=400, detail=f"Schema name '{request.schema_name}' already exists. Choose a different name.")
        schema_map[request.schema_name] = request.tables
        logger.info(f"Set schema '{request.schema_name}' with tables: {request.tables}")
        return {"message": f"Schema '{request.schema_name}' set successfully for file {request.file_name}"}
    except Exception as e:
        logger.error(f"Error setting schema: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error setting schema: {str(e)}")

@app.get("/excel-tables/")
async def get_excel_tables():
    try:
        tables = excel_processor.get_table_info()
        return {"tables": tables}
    except Exception as e:
        logger.error(f"Error getting table info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting table info: {str(e)}")

@app.post("/sql-query/")
async def sql_query(query: str = Query(...), schema_name: Optional[str] = Query(None)):
    logger.info(f"SQL query request: {query}, schema: {schema_name}")
    start_time = time.time()

    context = {"schema_name": schema_name} if schema_name else None
    cached_response = get_cached_response(query, "sql", context)
    if cached_response:
        logger.info(f"Cache hit for SQL query: {query}, schema: {schema_name}")
        end_time = time.time()
        logger.info(f"SQL Query Response Time (cached): {end_time - start_time:.2f} sec")
        return {"query": query, "response": cached_response}

    try:
        if schema_name and schema_name in schema_map:
            original_get_table_info = excel_processor.get_table_info
            def filtered_table_info():
                full_tables = original_get_table_info()
                filtered_tables = {k: v for k, v in full_tables.items() if k in schema_map[schema_name]}
                return filtered_tables
            excel_processor.get_table_info = filtered_table_info

        response = excel_processor.process_natural_language_query(query)

        if schema_name and schema_name in schema_map:
            excel_processor.get_table_info = original_get_table_info

        set_cached_response(query, response, "sql", context)

        end_time = time.time()
        logger.info(f"SQL Query Response Time: {end_time - start_time:.2f} sec")
        return {"query": query, "response": response}
    except Exception as e:
        logger.error(f"Error processing SQL query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing SQL query: {str(e)}")

@app.post("/web-search/")
async def web_search(question: str = Query(...)):
    logger.info(f"Web search request: {question}")
    start_time = time.time()

    cached_response = get_cached_response(question, "web")
    if cached_response:
        logger.info(f"Cache hit for web search: {question}")
        end_time = time.time()
        logger.info(f"Web Search Response Time (cached): {end_time - start_time:.2f} sec")
        return {"question": question, "answer": cached_response}

    try:
        web_agent = WebSearchAgent()
        answer = web_agent.search_web(question)
        
        set_cached_response(question, answer, "web")

        end_time = time.time()
        logger.info(f"Web Search Response Time: {end_time - start_time:.2f} sec")
        return {"question": question, "answer": answer}
    except Exception as e:
        logger.error(f"Error processing web search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing web search: {str(e)}")

@app.post("/query/")
async def query_rag(question: str = Query(...), sds_paths: Optional[str] = Query(None)):
    logger.info(f"Query request: {question}, sds_paths: {sds_paths}")
    start_time = time.time()

    paths = [sds_paths] if sds_paths else [str(f) for f in UPLOAD_DIR.glob("*.pdf")] + [str(f) for f in UPLOAD_DIR.glob("*.docx")]
    logger.info(f"Querying with {len(paths)} documents: {paths}")

    context = {"sds_paths": paths} if paths else None
    cached_response = get_cached_response(question, "document", context)
    if cached_response:
        logger.info(f"Cache hit for document query: {question}, paths: {paths}")
        end_time = time.time()
        logger.info(f"API Response Time (cached): {end_time - start_time:.2f} sec")
        return {"question": question, "answer": cached_response}

    if not paths:
        try:
            from app.rag import RAGSystem
            rag_system = RAGSystem([])
            if not rag_system.vectorstore or len(rag_system.vectorstore.docstore._dict) == 0:
                logger.warning("No data available for querying")
                raise HTTPException(status_code=400, detail="No data available. Upload files or submit URLs first.")
        except Exception as e:
            logger.error(f"Error accessing FAISS: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error accessing FAISS: {str(e)}")

    answer = run_agent_workflow(question, paths)
    
    set_cached_response(question, answer, "document", context)

    end_time = time.time()
    logger.info(f"API Response Time: {end_time - start_time:.2f} sec")

    return {"question": question, "answer": answer}

if __name__ == "__main__":
    logger.info(f"Starting {CONFIG['app']['name']} server")
    uvicorn.run(app, host=CONFIG["api"]["host"], port=CONFIG["api"]["port"])