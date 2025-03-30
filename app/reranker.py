from typing import List
from rank_bm25 import BM25Okapi

def rerank_chunks(chunks: List[str], query: str):
    if not chunks:
        print("Warning: No chunks retrieved for reranking.")
        return []

    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    scored_chunks = [(chunk, score) for chunk, score in zip(chunks, scores)]
    sorted_chunks = [chunk for chunk, _ in sorted(scored_chunks, key=lambda x: x[1], reverse=True)]
    return sorted_chunks[:3]
