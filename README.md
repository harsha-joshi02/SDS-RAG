# SDS-RAG: Retrieval-Augmented Generation System
SDS-RAG is a Retrieval-Augmented Generation (RAG) system that processes documents (PDF, DOCX) and URLs, retrieves relevant information, and answers user queries using a Large Language Model.


# Installation and Setup

1. Clone the Repository <br>
git clone https://github.com/your-username/your-repo.git <br>
cd your-repo <br>

2. Set Up a Virtual Environment <br>
python3 -m venv venv <br>
source venv/bin/activate <br>  

3. Install Dependencies <br>
pip install -r requirements.txt <br>

4. Set Up Environment Variables <br>
Create a .env file and add your Tavily API key: <br>


# Running the System

Open Terminal <br>
python main.py <br>

Open 2nd Terminal <br>
streamlit run frontend.py <br>


# Usage

1. Upload Documents <br>
Select PDFs or DOCX files and upload them. <br>
The system extracts text and indexes it using FAISS. <br>
2. Submit a URL <br>
Enter a website URL, and the system fetches its content. <br>
The retrieved text is indexed for future queries.<br>
3. Query the System <br>
Ask a question, and the system retrieves relevant chunks.<br>
BM25 reranks results, and the LLM generates a response.<br>


# API Endpoints

|Method	| Endpoint	    | Description <br>                                       |
|-------|---------------|--------------------------------------------------------|
|POST	| /upload-sds/	| Upload PDFs/DOCX files for indexing <br>               |
|POST	| /submit-url/	| Submit a URL for content extraction <br>               |
|POST	| /query/	    | Ask a question based on uploaded/processed content <br>|


# Dependencies

fastapi (API framework) <br>
uvicorn (ASGI server) <br>
streamlit (Frontend UI) <br>
faiss-cpu (Vector store) <br>
rank_bm25 (BM25 reranking) <br>
langchain (LLM interaction) <br>
tavily-python (Web search API) <br>
sentence-transformers (Embeddings) <br>
python-docx, pypdf (Document parsing) <br>
python-dotenv (Environment variables) <br>