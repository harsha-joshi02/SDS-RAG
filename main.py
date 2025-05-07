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
from app.graph import run_agent_workflow
from app.excel_processor import ExcelToSQLProcessor
from app.web_search import WebSearchAgent
from app.cache import get_cached_response, set_cached_response
from app.evaluation import EvaluationSystem

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
schema_map = {}
evaluator = EvaluationSystem()

UPLOAD_DIR = Path(CONFIG["app"]["upload_dir"])
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
    """
    Handles the upload of multiple SDS files and saves them to the server.

    Args:
        files (List[UploadFile]): A list of files to be uploaded.

    Returns:
        dict: A dictionary containing the paths of the uploaded files.

    Raises:
        HTTPException: If there is an error while saving the files.
    """

    file_paths = []
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())
        file_paths.append(str(file_path))
        logger.info(f"Uploaded file: {file.filename}")
    return {"file_paths": file_paths}

@app.post("/upload-excel/")
async def upload_excel(file: UploadFile = File(...)):
    """
    Handles the upload and processing of an Excel file.

    Args:
        file (UploadFile): The Excel file to be uploaded and processed.

    Returns:
        dict: A dictionary containing the name of the uploaded file, the tables created from it, and a success message.

    Raises:
        HTTPException: 
            - If the uploaded file is not an Excel file (not `.xlsx` or `.xls`).
            - If there is an error while saving or processing the file.
    """

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
    """
    Handles the setting of a new schema for a file.

    Args:
        request (SchemaRequest): The request body containing the schema name, tables, and file name.

    Returns:
        dict: A dictionary with a success message indicating the schema has been set.

    Raises:
        HTTPException:
            - If the schema name already exists, returns a 400 status with a message indicating the schema exists.
            - If there is an error while setting the schema, returns a 500 status with the error details.
    """

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
    """
    Fetches information about the tables available in the uploaded Excel files.

    Returns:
        dict: A dictionary containing the list of tables available in the system.

    Raises:
        HTTPException:
            - If an error occurs while retrieving table information, returns a 500 status with the error details.
    """

    try:
        tables = excel_processor.get_table_info()
        return {"tables": tables}
    except Exception as e:
        logger.error(f"Error getting table info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting table info: {str(e)}")

@app.post("/sql-query/")
async def sql_query(query: str = Query(...), schema_name: Optional[str] = Query(None), evaluate_metrics: bool = Query(False)):
    """
    Handles SQL query requests and retrieves the response by processing the query against the available schemas.

    Parameters:
        query (str): The SQL query to be executed.
        schema_name (str, optional): The name of the schema to filter the tables for the query. If not provided, all tables are used.
        evaluate_metrics (bool): Whether to evaluate and return metrics (e.g., hallucination, context precision) for the query. Default is False.

    Returns:
        dict: A dictionary containing the following:
            - "query": The original SQL query.
            - "response": The response generated from processing the query.
            - "metrics": Evaluation metrics (empty dictionary if `evaluate_metrics` is False or not implemented).

    Raises:
        HTTPException: If an error occurs while processing the query, returns a 500 status with error details.

    Cache Logic:
        If the query has been previously processed and cached, the cached response is returned to improve performance.
    """

    logger.info(f"SQL query request: {query}, schema: {schema_name}, evaluate_metrics: {evaluate_metrics}")
    start_time = time.time()

    context = {"schema_name": schema_name} if schema_name else None
    cached_response = get_cached_response(query, "sql", context)
    if cached_response:
        logger.info(f"Cache hit for SQL query: {query}, schema: {schema_name}")
        end_time = time.time()
        logger.info(f"SQL Query Response Time (cached): {end_time - start_time:.2f} sec")
        return {"query": query, "response": cached_response, "metrics": {}}

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

        metrics = {}
        if evaluate_metrics:
            logger.info("Evaluation metrics requested for SQL query, but no evaluation implemented yet")
            metrics = {"hallucination": 0.0, "context_precision": 0.0}

        set_cached_response(query, response, "sql", context)

        end_time = time.time()
        logger.info(f"SQL Query Response Time: {end_time - start_time:.2f} sec")
        return {"query": query, "response": response, "metrics": metrics}
    except Exception as e:
        logger.error(f"Error processing SQL query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing SQL query: {str(e)}")

@app.post("/web-search/")
async def web_search(question: str = Query(...), evaluate_metrics: bool = Query(False)):
    """
    Handles web search requests and retrieves the answer by querying the web.

    Parameters:
        question (str): The question to be searched on the web.
        evaluate_metrics (bool): Whether to evaluate and return metrics (e.g., hallucination, context precision) for the web search. Default is False.

    Returns:
        dict: A dictionary containing the following:
            - "question": The original question asked in the search.
            - "answer": The web search result or response generated.
            - "metrics": Evaluation metrics (empty dictionary if `evaluate_metrics` is False or not implemented).
            - "sources": A list of sources or URLs from where the information was fetched.

    Raises:
        HTTPException: If an error occurs while processing the web search, returns a 500 status with error details.

    Cache Logic:
        If the question has been previously processed and cached, the cached answer is returned to improve performance.
    """

    logger.info(f"Web search request: {question}, evaluate_metrics: {evaluate_metrics}")
    start_time = time.time()

    cached_response = get_cached_response(question, "web")
    if cached_response:
        logger.info(f"Cache hit for web search: {question}")
        end_time = time.time()
        logger.info(f"Web Search Response Time (cached): {end_time - start_time:.2f} sec")
        return {
            "question": question,
            "answer": cached_response,
            "metrics": {},
            "sources": []
        }

    try:
        web_agent = WebSearchAgent()
        result = web_agent.search_web(question)
        answer = result["answer"]
        ground_truth = result["ground_truth"]
        sources = result["sources"]
        
        metrics = {}
        if evaluate_metrics:
            metrics = evaluator.evaluate_response(question, answer, ground_truth, is_web_search=True)
            logger.info(f"Web search evaluation metrics: {metrics}")
        
        set_cached_response(question, answer, "web")

        end_time = time.time()
        logger.info(f"Web Search Response Time: {end_time - start_time:.2f} sec")
        return {
            "question": question,
            "answer": answer,
            "metrics": metrics,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error processing web search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing web search: {str(e)}")

@app.post("/query/")
async def query_rag(question: str = Query(...), sds_paths: Optional[str] = Query(None), evaluate_metrics: bool = Query(False)):
    """
    Handles document-based query requests for a Retrieval-Augmented Generation (RAG) system.

    Parameters:
        question (str): The question to be asked in the query.
        sds_paths (Optional[str]): A comma-separated list of paths to the documents (PDF, DOCX) to query. If not provided, all documents in the UPLOAD_DIR are used.
        evaluate_metrics (bool): Whether to evaluate and return metrics (e.g., hallucination, context precision) for the query response. Default is False.

    Returns:
        dict: A dictionary containing the following keys:
            - "question": The original query question.
            - "answer": The response generated for the query.
            - "metrics": Evaluation metrics for the query (empty dictionary if `evaluate_metrics` is False or not implemented).
            - "sources": A list of document paths or sources used to generate the answer.

    Raises:
        HTTPException: If an error occurs, such as when no files are uploaded or if an error occurs accessing the vector store, a 400 or 500 status code is returned with an error message.
    """

    logger.info(f"Query request: {question}, sds_paths: {sds_paths}, evaluate_metrics: {evaluate_metrics}")
    start_time = time.time()

    paths = [sds_paths] if sds_paths else [str(f) for f in UPLOAD_DIR.glob("*.pdf")] + [str(f) for f in UPLOAD_DIR.glob("*.docx")]
    logger.info(f"Querying with {len(paths)} documents: {paths}")

    context = {"sds_paths": paths} if paths else None
    cached_response = get_cached_response(question, "document", context)
    if cached_response:
        logger.info(f"Cache hit for document query: {question}, paths: {paths}")
        end_time = time.time()
        logger.info(f"API Response Time (cached): {end_time - start_time:.2f} sec")
        return {
            "question": question,
            "answer": cached_response,
            "metrics": {},
            "sources": []
        }

    if not paths:
        try:
            from app.rag import RAGSystem
            rag_system = RAGSystem([])
            if not rag_system.vectorstore or len(rag_system.vectorstore.docstore._dict) == 0:
                logger.warning("No data available for querying")
                raise HTTPException(status_code=400, detail="No data available. Upload files first.")
        except Exception as e:
            logger.error(f"Error accessing FAISS: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error accessing FAISS: {str(e)}")

    result = run_agent_workflow(question, paths, evaluate_metrics=evaluate_metrics)
    
    logger.info(f"Document query evaluation metrics: {result['metrics']}")
    
    set_cached_response(question, result["answer"], "document", context)

    end_time = time.time()
    logger.info(f"API Response Time: {end_time - start_time:.2f} sec")

    return {
        "question": question,
        "answer": result["answer"],
        "metrics": result["metrics"],
        "sources": result.get("sources", [])
    }

if __name__ == "__main__":
    logger.info(f"Starting {CONFIG['app']['name']} server")
    uvicorn.run(app, host=CONFIG["api"]["host"], port=CONFIG["api"]["port"])