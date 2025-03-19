cache = {}

def get_cached_response(query: str):
    return cache.get(query)

def set_cached_response(query: str, response: str):
    cache[query] = response