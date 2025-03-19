from typing import List
import os
import logging
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.utils import load_sds, preprocess_text
from app.cache import get_cached_response, set_cached_response
from app.reranker import rerank_chunks
from app.formatter import format_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, sds_paths: List[str]):
        self.sds_paths = sds_paths
        self.llm = ChatOllama(model="llama3.2", temperature=0.5)
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = self._load_and_index_sds()
        self.vectorstore.save_local("./faiss_index")

    def _load_and_index_sds(self):
        all_chunks = []
        for path in self.sds_paths:
            try:
                text = load_sds(path)
                chunks = preprocess_text(text)
                chunks_with_meta = [{"text": chunk, "source": os.path.basename(path)} for chunk in chunks]
                all_chunks.extend(chunks_with_meta)
                logger.info(f"Loaded {path} with {len(chunks)} chunks")
            except Exception as e:
                logger.warning(f"Skipped {path} due to error: {str(e)}")
                continue

        if not all_chunks:
            logger.warning("No chunks loaded, creating empty index")
            return FAISS.from_texts([""], self.embedding_model)
        
        texts = [chunk["text"] for chunk in all_chunks]
        metadatas = [{"source": chunk["source"]} for chunk in all_chunks]

        try:
            vectorstore = FAISS.load_local("./faiss_index", self.embedding_model, allow_dangerous_deserialization=True)
            vectorstore.add_texts(texts, metadatas=metadatas)
            logger.info("Updated existing FAISS index")
        except:
            vectorstore = FAISS.from_texts(texts, self.embedding_model, metadatas=metadatas)
            logger.info("Created new FAISS index")
            
        return vectorstore

    def query(self, question: str):
        cached = get_cached_response(question)
        if cached:
            return cached
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(question)
        chunk_texts = [doc.page_content for doc in docs]
        chunk_metadatas = [doc.metadata for doc in docs]
        reranked_chunks = rerank_chunks(chunk_texts, question)
        context = "\n".join(reranked_chunks)
        prompt = f"Query: {question}\nContext: {context}\nAnswer:"
        result = self.llm.invoke(prompt)
        answer = result.content
        formatted_answer = format_response(answer, reranked_chunks, chunk_metadatas)

        set_cached_response(question, formatted_answer)
        return formatted_answer

    def add_document(self, text: str):
        chunks = preprocess_text(text)
        chunks_with_meta = [{"text": chunk, "source": "web_content"} for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks_with_meta]
        metadatas = [{"source": chunk["source"]} for chunk in chunks_with_meta]
        self.vectorstore.add_texts(texts, metadatas=metadatas)
        self.vectorstore.save_local("./faiss_index")