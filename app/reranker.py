from typing import List

def rerank_chunks(chunks: List[str], query: str):
    scored = [(chunk, len(chunk)) for chunk in chunks]
    sorted_chunks = [chunk for chunk, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
    return sorted_chunks[:3]