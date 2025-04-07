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
