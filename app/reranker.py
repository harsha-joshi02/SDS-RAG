from typing import List
from rank_bm25 import BM25Okapi
import numpy as np

def rerank_chunks(chunks: List[str], query: str, k: int = 3):
    if not chunks:
        print("Warning: No chunks retrieved for reranking.")
        return []

    if not query.strip():
        print("Warning: Empty query. Returning original chunks.")
        return chunks[:k]

    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    if not any(scores):
        print("Warning: BM25 scores are all zero. Query may not match chunks well.")

    scores = np.array(scores)
    if scores.max() > 0:
        scores = scores / scores.max()

    top_indices = np.argsort(scores)[::-1][:k]
    top_chunks = [chunks[i] for i in top_indices]

    for i, idx in enumerate(top_indices):
        print(f"Top-{i+1} Chunk (Score: {scores[idx]:.4f}): {chunks[idx][:100]}...")

    return top_chunks
