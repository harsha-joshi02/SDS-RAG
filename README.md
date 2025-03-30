# SDS-RAG: Retrieval-Augmented Generation System
SDS-RAG is a Retrieval-Augmented Generation (RAG) system that processes documents (PDF, DOCX) and URLs, retrieves relevant information, and answers user queries using a Large Language Model (LLM).

1. Document Upload: Extracts and indexes content from PDFs and DOCX files
2. Web Search Integration: Retrieves content from URLs using the Tavily API
3. Efficient Retrieval: Uses FAISS for vector storage and BM25 for reranking
4. LLM-Powered Responses: Generates responses based on retrieved documents


# Installation and Setup

1. Clone the Repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

2. Set Up a Virtual Environment
python3 -m venv venv
source venv/bin/activate  

3. Install Dependencies
pip install -r requirements.txt

4. Set Up Environment Variables
Create a .env file and add your Tavily API key:


# Running the System

Open Terminal
python main.py

Open 2nd Terminal
streamlit run frontend.py


# Usage

1. Upload Documents
Select PDFs or DOCX files and upload them.
The system extracts text and indexes it using FAISS.
2. Submit a URL
Enter a website URL, and the system fetches its content.
The retrieved text is indexed for future queries.
3. Query the System
Ask a question, and the system retrieves relevant chunks.
BM25 reranks results, and the LLM generates a response.


# API Endpoints

Method	Endpoint	    Description
POST	/upload-sds/	Upload PDFs/DOCX files for indexing
POST	/submit-url/	Submit a URL for content extraction
POST	/query/	        Ask a question based on uploaded/processed content


# Project Structure

rag-system/
│── app/
│   ├── __init__.py
│   ├── cache.py           # Implements caching
│   ├── formatter.py       # Formats responses with citations
│   ├── prompt_template.py # LLM prompt template
│   ├── rag.py             # Main RAG logic (retrieval, reranking, generation)
│   ├── reranker.py        # BM25 reranking implementation
│   ├── search.py          # Handles URL-based retrieval
│   ├── utils.py           # File loading and preprocessing functions
│── data/                  # Directory for uploaded documents
│── frontend.py            # Streamlit UI
│── main.py                # FastAPI backend
│── requirements.txt       # Python dependencies
│── .env                   # API keys and environment variables
│── .gitignore             # Files to ignore in version control
│── README.md              # Project documentation


# Dependencies

fastapi (API framework)
uvicorn (ASGI server)
streamlit (Frontend UI)
faiss-cpu (Vector store)
rank_bm25 (BM25 reranking)
langchain (LLM interaction)
tavily-python (Web search API)
sentence-transformers (Embeddings)
python-docx, pypdf (Document parsing)
python-dotenv (Environment variables)