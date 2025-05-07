from typing import List
from rank_bm25 import BM25Okapi
import numpy as np
import logging
from app.config import CONFIG

logger = logging.getLogger(__name__)

def rerank_chunks(chunks: List[str], query: str, k: int = CONFIG["reranker"]["top_k"]):
    """
    Reranks document chunks based on relevance to the query using BM25.

    This function tokenizes the chunks and the query, calculates BM25 scores, and returns the top-k most relevant chunks.

    Parameters:
        chunks (List[str]): The list of document chunks to be ranked.
        query (str): The query to compare against the chunks.
        k (int): The number of top chunks to return (default is CONFIG["reranker"]["top_k"]).

    Returns:
        List[str]: The top-k most relevant chunks.

    Logs warnings if no valid chunks or tokens are found, or if BM25 scores are low.
    """

    if not chunks or all(not chunk.strip() for chunk in chunks):
        logger.warning("No valid chunks retrieved for reranking.")
        return []

    if not query.strip():
        logger.warning("Empty query. Returning original chunks.")
        return chunks[:k]

    tokenized_chunks = [chunk.split() for chunk in chunks if chunk.strip()]
    
    if not tokenized_chunks:
        logger.warning("No valid tokenized chunks after processing.")
        return []

    for i, tokens in enumerate(tokenized_chunks[:3]):
        logger.info(f"Chunk {i} token count: {len(tokens)}")
        if len(tokens) < 5:
            logger.warning(f"Very few tokens in chunk {i}: {tokens}")
    
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = query.split()
    logger.info(f"Query tokens: {tokenized_query}")
    
    scores = bm25.get_scores(tokenized_query)

    if not any(scores):
        logger.warning("BM25 scores are all zero. Query may not match chunks well.")

    scores = np.array(scores)
    if scores.max() > 0:
        scores = scores / scores.max()

    top_indices = np.argsort(scores)[::-1][:k]
    top_chunks = [chunks[i] for i in top_indices]

    for i, idx in enumerate(top_indices):
        logger.info(f"Top-{i+1} Chunk (Score: {scores[idx]:.4f}): {chunks[idx][:100]}...")

    return top_chunks