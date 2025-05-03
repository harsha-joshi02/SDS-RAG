import logging

logger = logging.getLogger(__name__)

def format_response(answer, chunks, metadatas):
    logger.info(f"Formatting response with {len(chunks)} chunks and {len(metadatas)} metadatas")
    
    if len(metadatas) < len(chunks):
        logger.warning(f"Metadata length ({len(metadatas)}) is less than chunks length ({len(chunks)})")
        metadatas.extend([{"source": "Unknown Source"} for _ in range(len(chunks) - len(metadatas))])
    
    citations = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        try:
            source = meta.get('source', 'Unknown Source')
            chunk_preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
            citation = f"[Doc {i+1}] ({source}): {chunk_preview}"
            citations.append(citation)
        except Exception as e:
            logger.error(f"Error formatting citation {i}: {str(e)}")
            citations.append(f"[Doc {i+1}] (Error formatting citation)")

    citations_text = "\n".join(citations)
    
    if citations:
        formatted_response = f"Answer: {answer}\n\nCitations:\n{citations_text}"
    else:
        formatted_response = answer
    
    logger.info(f"Final formatted response length: {len(formatted_response)}")
    return formatted_response