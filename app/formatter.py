from typing import List

def format_response(response: str, chunks: List[str], metadatas: List[dict]):
    citations = "\n".join([f"[Doc {i+1}] ({meta['source']}): {chunk[:50]}..." for i, (chunk, meta) in enumerate(zip(chunks, metadatas))])
    return f"{response}\n\n**Sources:**\n{citations}"