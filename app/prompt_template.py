from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["retrieved_documents", "query"],
    template=(
        "You are an AI assistant that answers questions based **only** on the provided documents.\n"
        "If the answer is explicitly mentioned in the given text, respond with the relevant details.\n"
        "If the answer is not found in the provided text, say: "
        "'The answer is not present in the given documents.'\n"
        "Do not attempt to answer from general knowledge or assumptions.\n\n"
        "**Context:**\n{retrieved_documents}\n\n"
        "**User Query:** {query}\n\n"
        "**Answer:**"
    ),
)
