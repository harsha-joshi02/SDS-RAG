from typing import List
import os
import time
import logging
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.utils import load_sds, preprocess_text
from app.cache import get_cached_response, set_cached_response
from app.reranker import rerank_chunks
from app.formatter import format_response
from app.prompt_template import prompt_template

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, sds_paths: List[str]):
        self.sds_paths = sds_paths
        self.llm = ChatOllama(model="llama3.2", temperature=0.0)
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        if os.path.exists("./faiss_index"):
            print("Loading existing FAISS index...")
            self.vectorstore = FAISS.load_local(
                "./faiss_index", 
                self.embedding_model, 
                allow_dangerous_deserialization=True  
            )
        else:
            print("No FAISS index found, creating a new one...")
            self.vectorstore = self._load_and_index_sds()

    def _load_and_index_sds(self):
        all_chunks = []
        for path in self.sds_paths:
            try:
                text = load_sds(path)
                chunks = preprocess_text(text)
                chunks_with_meta = [{"text": chunk, "source": os.path.basename(path)} for chunk in chunks]
                all_chunks.extend(chunks_with_meta)
            except Exception as e:
                logger.warning(f"Skipped {path} due to error: {str(e)}")
                continue

        if not all_chunks:
            logger.warning("No chunks loaded, creating empty index")
            return FAISS.from_texts([""], self.embedding_model)

        texts = [chunk["text"] for chunk in all_chunks]
        metadatas = [{"source": chunk["source"]} for chunk in all_chunks]

        vectorstore = FAISS.from_texts(texts, self.embedding_model, metadatas=metadatas)
        return vectorstore

    def query(self, question: str):
        start_time = time.time()
        cached = get_cached_response(question)
        if cached:
            return cached

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        docs = retriever.get_relevant_documents(question)
        chunk_texts = [doc.page_content for doc in docs]
        chunk_metadatas = [doc.metadata for doc in docs]

        if not chunk_texts:
            return "The answer is not present in the given documents."

        reranked_chunks = rerank_chunks(chunk_texts, question)
        context = "\n".join(reranked_chunks)

        formatted_prompt = prompt_template.format(
            retrieved_documents=context, query=question
        )

        result = self.llm.invoke(formatted_prompt)
        end_time = time.time()
        print(f"Total Query Time: {end_time - start_time:.2f} sec")

        answer = result.content
        formatted_answer = format_response(answer, reranked_chunks, chunk_metadatas)

        set_cached_response(question, formatted_answer)

        return formatted_answer

    def add_document(self, text: str):
        start_time = time.time()
        print(f"Adding document, text length: {len(text)}")
        chunks = preprocess_text(text)
        print(f"Chunks created: {len(chunks)}")
        if not chunks:
            logger.warning("No chunks created from document")
            return
        chunks_with_meta = [{"text": chunk, "source": "web_content"} for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks_with_meta]
        metadatas = [{"source": chunk["source"]} for chunk in chunks_with_meta]
        print(f"Adding {len(texts)} chunks to vectorstore")
        self.vectorstore.add_texts(texts, metadatas=metadatas)

        end_time = time.time()
        print(f"FAISS Indexing Time: {end_time - start_time:.2f} sec")
        logger.info(f"Loaded web_content with {len(chunks)} chunks")
        self.vectorstore.save_local("./faiss_index")
