from tavily import TavilyClient
from app.rag import RAGSystem

client = TavilyClient(api_key="")

def search_url(url: str):
    response = client.get_search_context(url, max_results=1)
    text = response if isinstance(response, str) else response.get("content", "")
    rag_system = RAGSystem([])
    rag_system.add_document(text)