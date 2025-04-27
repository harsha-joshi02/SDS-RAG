import time
from app.config import CONFIG

cache = {}

def get_cached_response(query: str):
    entry = cache.get(query)
    if entry:
        response, timestamp = entry
        if time.time() - timestamp < CONFIG["cache"]["ttl_seconds"]:
            return response
        else:
            del cache[query] 
    return None

def set_cached_response(query: str, response: str):
    cache[query] = (response, time.time())

# faithfullness
# hallucination
# answer_relevance
# LangSmith
# JSON OutputParser (Pydantic) 

# Agent: To interact with SQL databases 
# Excel -> SQL table -> User query (natural language) -> SQL query -> retrieve answer
# Drop down in the query tab. So that user can chat with the documents separately
# Memory - Redis, LangChain
# Multiple Tables