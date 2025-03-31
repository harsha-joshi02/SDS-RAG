def format_response(answer, chunks, metadatas):
    print("Chunk Metadatas:", metadatas)
    citations = "\n".join([
        f"[Doc {i+1}] ({meta.get('source', 'Unknown Source')}): {chunk[:50]}..."
        for i, (chunk, meta) in enumerate(zip(chunks, metadatas))
    ])
    formatted_response = f"Answer: {answer}\n\nCitations:\n{citations}" if citations else answer
    return formatted_response
