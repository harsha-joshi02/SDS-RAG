from typing import List
import logging
from app.rag import RAGSystem
from app.web_search import WebSearchAgent
from app.config import CONFIG

logger = logging.getLogger(__name__)

class AgentCoordinator:
    def __init__(self, sds_paths: List[str]):
        self.doc_agent = RAGSystem(sds_paths)
        self.web_agent = WebSearchAgent()
        self.confidence_threshold = CONFIG["agent"]["confidence_threshold"]

    def query(self, question: str) -> str:
        logger.info(f"Coordinating query: {question}")

        doc_answer, confidence = self.doc_agent.query(question)
        logger.info(f"Document Agent confidence: {confidence:.2f}")

        if confidence >= self.confidence_threshold:
            logger.info("Using Document Agent response")
            return doc_answer

        logger.info("Falling back to Web Search Agent")
        web_answer = self.web_agent.search_web(question)
        return web_answer