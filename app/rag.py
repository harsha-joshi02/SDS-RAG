from typing import List, Tuple
import os
import time
import logging
import json
from rank_bm25 import BM25Okapi
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.utils import load_sds, preprocess_text
from app.cache import get_cached_response, set_cached_response
from app.reranker import rerank_chunks
from app.formatter import format_response
from app.prompt_template import doc_prompt_template
from dotenv import load_dotenv
from app.config import CONFIG

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, sds_paths: List[str]):
        self.sds_paths = sds_paths
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.embedding_model = HuggingFaceEmbeddings(model_name=CONFIG["embedding"]["model_name"])
        self.vectorstore = self._load_and_index_sds()
        logger.info(f"Initialized RAGSystem with {len(self.sds_paths)} paths: {self.sds_paths}")

    def _load_and_index_sds(self):
        all_chunks = []
        for path in self.sds_paths:
            try:
                text = load_sds(path)
                chunks = preprocess_text(text)
                source = os.path.basename(path)
                chunks_with_meta = [{"text": chunk, "source": source} for chunk in chunks]
                all_chunks.extend(chunks_with_meta)
                logger.info(f"Loaded {len(chunks)} chunks from {path}")
            except Exception as e:
                logger.warning(f"Skipped {path} due to error: {str(e)}")
                continue

        if not all_chunks:
            logger.warning("No chunks loaded, creating empty index")
            return FAISS.from_texts([""], self.embedding_model)

        texts = [chunk["text"] for chunk in all_chunks]
        metadatas = [{"source": chunk["source"]} for chunk in all_chunks]

        vectorstore = FAISS.from_texts(texts, self.embedding_model, metadatas=metadatas)
        logger.info(f"Created FAISS index with {len(texts)} chunks")
        return vectorstore

    def query(self, question: str) -> Tuple[str, float]:
        start_time = time.time()
        # Check cache using query_type='document' and context with sds_paths
        cached = get_cached_response(question, query_type="document", context={"sds_paths": self.sds_paths})
        if cached:
            logger.info(f"Cache hit for question: '{question}' with paths: {self.sds_paths}")
            return cached, 1.0 

        logger.info(f"Cache miss for question: '{question}' with paths: {self.sds_paths}")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": CONFIG["retriever"]["search_k"]})

        docs = retriever.get_relevant_documents(question)
        expected_sources = [os.path.basename(path) for path in self.sds_paths]
        filtered_docs = [doc for doc in docs if doc.metadata.get("source") in expected_sources]
        
        chunk_texts = [doc.page_content for doc in filtered_docs]
        chunk_metadatas = [doc.metadata for doc in filtered_docs]
        
        logger.info(f"Retrieved {len(docs)} documents, filtered to {len(filtered_docs)} from sources: {expected_sources}")
        
        for i, doc in enumerate(filtered_docs[:3]):
            logger.info(f"Filtered Document {i+1} (source: {doc.metadata.get('source')}): {doc.page_content[:100]}...")

        if not filtered_docs:
            logger.warning("No relevant documents found for the query in the specified sources")
            answer = "The answer is not present in the given documents."
            set_cached_response(question, answer, query_type="document", context={"sds_paths": self.sds_paths})
            return answer, 0.0

        reranked_chunks = rerank_chunks(chunk_texts, question)
        if not reranked_chunks:
            logger.warning("No relevant chunks found after reranking")
            answer = "The answer is not present in the given documents."
            set_cached_response(question, answer, query_type="document", context={"sds_paths": self.sds_paths})
            return answer, 0.0

        context = "\n".join(reranked_chunks)

        tokenized_chunks = [chunk.split() for chunk in chunk_texts]
        bm25 = BM25Okapi(tokenized_chunks)
        tokenized_query = question.split()
        scores = bm25.get_scores(tokenized_query)
        confidence = max(scores) / (max(scores) + 1) if max(scores) > 0 else 0.0  

        formatted_prompt = doc_prompt_template.format(
            retrieved_documents=context, query=question
        )

        logger.info(f"Sending prompt to Groq API, length: {len(formatted_prompt)}")
        response = self.client.chat.completions.create(
            model=CONFIG["groq"]["model"], 
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=CONFIG["groq"]["temperature"],
            max_tokens=CONFIG["groq"]["max_tokens"]
        )
        end_time = time.time()
        logger.info(f"Total Query Time: {end_time - start_time:.2f} sec")

        answer = response.choices[0].message.content
        formatted_answer = format_response(answer, reranked_chunks, chunk_metadatas)

        set_cached_response(question, formatted_answer, query_type="document", context={"sds_paths": self.sds_paths})

        return formatted_answer, confidence

    def add_document(self, text: str):
        start_time = time.time()
        logger.info(f"Adding document, text length: {len(text)}")
        
        try:
            if text.strip().startswith('[') and text.strip().endswith(']'):
                json_data = json.loads(text)
                if isinstance(json_data, list) and len(json_data) > 0 and isinstance(json_data[0], dict):
                    if "content" in json_data[0]:
                        text = json_data[0]["content"]
                        logger.info(f"Extracted content from JSON array, new length: {len(text)}")
            elif text.strip().startswith('{') and text.strip().endswith('}'):
                json_data = json.loads(text)
                if isinstance(json_data, dict) and "content" in json_data:
                    text = json_data["content"]
                    logger.info(f"Extracted content from JSON object, new length: {len(text)}")
        except json.JSONDecodeError:
            logger.warning("Failed to parse text as JSON, continuing with original text")
        except Exception as e:
            logger.warning(f"Error processing text: {str(e)}")
        
        chunks = preprocess_text(text)
        logger.info(f"Chunks created: {len(chunks)}")
        
        if not chunks:
            logger.warning("No chunks created from document")
            return

        chunks_with_meta = [{"text": chunk, "source": "web_content"} for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks_with_meta]
        metadatas = [{"source": chunk["source"]} for chunk in chunks_with_meta]

        logger.info(f"Adding {len(texts)} chunks to vectorstore")
        self.vectorstore.add_texts(texts, metadatas=metadatas)

        logger.info(f"Added web_content with {len(chunks)} chunks to temporary index")
        end_time = time.time()
        logger.info(f"Indexing Time: {end_time - start_time:.2f} sec")