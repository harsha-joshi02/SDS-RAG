from tavily import TavilyClient
from app.rag import RAGSystem
from dotenv import load_dotenv
import os
import time

load_dotenv()
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_url(url: str):
    start_time = time.time()
    print(f"Submitting URL: {url}")
    response = client.get_search_context(url, max_results=1)
    text = response if isinstance(response, str) else response.get("content", "")
    end_time = time.time()
    print(f"URL Processing Time: {end_time - start_time:.2f} sec")
    rag_system = RAGSystem([])
    rag_system.add_document(text)