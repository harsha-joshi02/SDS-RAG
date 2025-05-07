# Retrieval-Augmented Generation System

This is a Retrieval-Augmented Generation (RAG) system designed to process documents (PDF, DOCX), Excel files, and web content, retrieve relevant information, and answer user queries using a Large Language Model (LLM). The system supports document indexing, SQL query translation for Excel data, and web searches, with a Streamlit-based frontend for user interaction.

# Features

* **Document Processing**: Upload and index PDF/DOCX files for retrieval using FAISS and BM25 reranking.
* **Excel to SQL**: Convert Excel sheets to SQLite tables and query them using natural language, translated to SQL.
* **Web Search**: Fetch and process web content for queries using the Tavily API.
* **Caching**: Cache responses for document, SQL, and web queries to improve performance.
* **Evaluation**: Evaluate responses with metrics like hallucination and context precision using Opik.
* **Frontend**: Interactive Streamlit interface for uploading files, submitting schemas, and querying the system.

# Installation and Setup

1. **Clone the Repository** <br>
`git clone https://github.com/your-username/your-repo.git` <br>
`cd repo` <br>

2. **Set Up a Virtual Environment** <br>
`python3 -m venv venv` <br>
MacOS: `source venv/bin/activate` <br>
Windows: `venv/Scripts/activate` <br>  

3. **Install Dependencies** <br>
`pip install -r requirements.txt` <br>

4. **Set Up Environment Variables** <br>
Create a .env file in the project root and add the following: <br>
    * TAVILY_API_KEY = tavily-api-key      (Obtain from https://tavily.com) <br>
    * GROQ_API_KEY = groq-api-key      (Obtain from https://console.groq.com/home) <br>
    * API_URL = http://localhost:8000    
    * OPIK_API_KEY = opik-api-key      (Obtain from https://www.comet.com/site/products/opik/) <br>

5. **Configuration** <br>
Ensure the config.yaml file is properly configured. Default settings include: <br>
    * FAISS index path: ./faiss_index <br>
    * Upload directory: data <br>
    * Cache TTL: 3600 seconds <br>
    * Groq model: llama-3.3-70b-versatile <br>
    * Embedding model: all-MiniLM-L6-v2 <br>

# Running the System

1. **Clean Up Previous Data (Optional)** <br>
`make cleanup` <br>

2. **Run the Backend** <br>
`make run_backend` <br>
(This starts the FastAPI server on http://localhost:8000) <br>

3. **Run the Frontend** <br>
`make run_frontend` <br>

4. **Run Both (Backend and Frontend)** <br>
`make all` <br>
(This performs cleanup, starts the backend, and launches the frontend) <br>


# Usage

1. **Upload Documents** <br>
    * Use the Streamlit interface to upload PDF or DOCX files. <br>
    * Files are saved in the data directory and indexed using FAISS for retrieval. <br>

2. **Upload Excel Files** <br>
    * Upload Excel files (.xlsx, .xls) containing tables. <br>
    * Each sheet is converted to a SQLite table in excel_data.sqlite <br>
    * Assign a schema name to group tables for querying. <br>

3. **Query the System** <br>
    * Document Query: Select a document and ask questions. The system retrieves relevant chunks, reranks them with BM25, and generates answers using the LLM. <br>
    * SQL Query: Select a schema and ask questions in natural language. The system translates queries to SQL and executes them on the SQLite database. <br>
    * Web Search: Choose "Web Search" to query web content via Tavily. Results are processed and answered by the LLM. <br>
    * Toggle "Show Evaluation Metrics" to view hallucination and context precision scores. <br>

4. **View Results** <br>
    * Answers include citations for document queries and web sources for web searches. <br>
    * Evaluation metrics are displayed in an expander if enabled. <br>
    * Chat history is maintained separately for document, SQL, and web queries. <br>


# API Endpoints

|Method	|    Endpoints	   | Description <br>                                                               |
|-------|------------------|--------------------------------------------------------------------------------|
|POST	| /upload-sds/	   | Upload PDFs/DOCX files for indexing <br>                                       |
|POST	| /upload-excel/   | Upload Excel files and convert sheets to SQLite tables. <br>                   |
|POST	| /set-schema/	   | Assign a schema name to group Excel tables for querying. <br>                  |
|GET    | /excel-tables/   | Retrieve information about SQLite tables created from Excel files. <br>        |
|POST   | /query/          | Query indexed documents with a question and optional document paths. <br>      |
|POST   | /sql-query/      | Query SQLite tables using natural language with an optional schema name. <br>  |
|POST   | /web-search/     | Perform a web search and answer a question based on web content. <br>          |

# Project Structure
* **app/**: Core application logic
    - **cache.py**: Response caching for document, SQL, and web queries.
    - **config.py**: Load configuration from config.yaml.
    - **excel_processor.py**: Convert Excel files to SQLite and handle SQL queries.
    - **formatter.py**: Format responses with citations.
    - **graph.py**: LangGraph workflow for coordinating document and web searches.
    - **prompt_template.py**: Prompt templates for document and web queries.
    - **rag.py**: RAG system for document indexing and querying.
    - **reranker.py**: BM25-based chunk reranking.
    - **search.py**: URL content extraction (not used in current frontend).
    - **utils.py**: Document loading and text preprocessing.
    - **web_search.py**: Web search using Tavily API.
    - **evaluation.py**: Response evaluation with Opik metrics.
* **data/**: Directory for uploaded files.
* **evaluations/**: Directory for evaluation results.
* **frontend.py**: Streamlit frontend for user interaction.
* **main.py**: FastAPI backend server.
* **config.yaml**: Configuration file.
* **cleanup.py**: Script to clear data and indexes.
* **Makefile**: Commands for running and cleaning the system.


# Dependencies

* **faiss-cpu**  (vector store for similarity search in RAG pipelines)  <br>
* **fastapi**  (backend web framework for building APIs)  <br>
* **groq**  (LLM API provider used for generating responses)  <br>
* **langchain**  (framework for building LLM-based applications)  <br>
* **langchain-community**  (community-contributed integrations for LangChain)  <br>
* **langgraph**  (stateful, agentic orchestration framework built on LangChain)  <br>
* **litellm**  (unified interface for using various LLM providers)  <br>
* **numpy**  (numerical computing library; used for array manipulation)  <br>
* **opik** (evaluation framework for checking hallucination, faithfulness, etc.)  <br>
* **pandas**  (data manipulation and analysis library; especially for tables/Excel)  <br>
* **pdfplumber**  (extracts text and metadata from PDF files)  <br>
* **python-docx**  (extracts text and formatting from DOCX files)  <br>
* **python-dotenv**  (loads environment variables from a .env file)  <br>
* **python-multipart**  (required by FastAPI to handle file uploads via form data)  <br>
* **pyyaml**  (parses and writes YAML files)  <br>
* **rank_bm25**  (implements BM25 algorithm for keyword-based text retrieval)  <br>
* **sentence-transformers**  (embeddings model used for semantic search in RAG)  <br>
* **streamlit**  (frontend framework for interactive web apps)  <br>
* **tavily-python**  (API client for web search using Tavily)  <br>
* **uvicorn**  (ASGI server to run FastAPI applications)  <br>
