import time

cache = {}

def get_cached_response(query: str):
    entry = cache.get(query)
    if entry:
        response, timestamp = entry
        if time.time() - timestamp < 3600:
            return response
        else:
            del cache[query] 
    return None

def set_cached_response(query: str, response: str):
    cache[query] = (response, time.time())
