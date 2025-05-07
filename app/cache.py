import time
from app.config import CONFIG

cache = {}

def get_cached_response(query: str, query_type: str, context: dict = None) -> str:
    """
    Retrieves a cached response for a given query if it exists and hasn't expired.

    Args:
        query (str): The input query string.
        query_type (str): The category or type of the query (e.g., 'search', 'document').
        context (dict, optional): Additional context to differentiate cache keys.

    Returns:
        str or None: The cached response if valid and not expired; otherwise, None.
    """
    cache_key = _create_cache_key(query, query_type, context)
        
    entry = cache.get(cache_key)
    if entry:
        response, timestamp = entry
        if time.time() - timestamp < CONFIG["cache"]["ttl_seconds"]:
            return response
        else:
            del cache[cache_key] 
    return None

def set_cached_response(query: str, response: str, query_type: str, context: dict = None):
    cache_key = _create_cache_key(query, query_type, context)
    cache[cache_key] = (response, time.time())

def _create_cache_key(query: str, query_type: str, context: dict = None) -> str:
    """
    Generates a unique cache key based on the query, its type, and optional context.

    Args:
        query (str): The input query string.
        query_type (str): The category of the query ('document', 'sql', or 'web').
        context (dict, optional): Additional context used to construct the cache key,
                                such as SDS paths for documents or schema name for SQL.

    Returns:
        str: A uniquely formatted string to be used as a cache key.
        
    Raises:
        ValueError: If an unsupported query_type is provided.
    """

    if query_type not in ['document', 'sql', 'web']:
        raise ValueError(f"Invalid query_type: {query_type}")

    if query_type == 'document' and context and 'sds_paths' in context:
        paths = ':'.join(sorted(context['sds_paths']))
        return f"{query_type}:{query}:{paths}"
    elif query_type == 'sql' and context and 'schema_name' in context:
        return f"{query_type}:{query}:{context['schema_name']}"
    elif query_type == 'web':
        return f"{query_type}:{query}"
    else:
        return f"{query_type}:{query}"