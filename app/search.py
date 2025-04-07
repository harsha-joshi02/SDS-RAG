from tavily import TavilyClient
from app.rag import RAGSystem
from dotenv import load_dotenv
import os
import time
import logging

logger = logging.getLogger(__name__)

load_dotenv()
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_url(url: str):
    start_time = time.time()
    logger.info(f"Submitting URL: {url}")
    
    try:

        response = client.extract(urls=[url], include_images=False, extract_depth="basic")
        
        if "results" in response and len(response["results"]) > 0:
            text = response["results"][0].get("raw_content", "")
        else:
            text = ""
            logger.warning("No content extracted from the URL.")
        
        logger.info(f"Extracted content length: {len(text)}")
        
        logger.info(f"Content preview: {text[:200]}...")
        
        end_time = time.time()
        logger.info(f"URL Processing Time: {end_time - start_time:.2f} sec")
        
        rag_system = RAGSystem([])
        rag_system.add_document(text)
        return {"success": True, "content_length": len(text)}
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}
