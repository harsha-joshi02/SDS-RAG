import logging
from typing import List, Dict, Any
from opik import Opik #type: ignore
import json
import os
from app.config import CONFIG
from app.rag import RAGSystem
from datetime import datetime
import opik #type: ignore
import importlib 
import inspect
from sentence_transformers import SentenceTransformer, util
import torch
from collections import Counter

logger = logging.getLogger(__name__)

class EvaluationSystem:
    def __init__(self):
        try:
            self.opik_client = Opik(api_key=os.getenv("OPIK_API_KEY"))
            self.output_dir = CONFIG["evaluation"]["output_dir"]
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Initialized Opik client (version: {opik.__version__})")
            
            try:
                import litellm #type: ignore
                self.model_name = CONFIG["groq"]["model"]
                litellm.groq_api_key = os.getenv("GROQ_API_KEY")
                litellm.model = f"groq/{self.model_name}"
                logger.info(f"Configured LiteLLM to use Groq model: {self.model_name}")
                test_response = litellm.completion(
                    model=f"groq/{self.model_name}",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=10
                )
                logger.info(f"LiteLLM Groq test successful: {test_response.choices[0].message.content[:50]}...")
            except Exception as e:
                logger.error(f"Failed to configure LiteLLM for Groq: {str(e)}")
                logger.warning("Metrics may fail unless Groq configuration is resolved.")
            
            try:
                self.similarity_model = SentenceTransformer(CONFIG["embedding"]["model_name"])
                logger.info(f"Initialized SentenceTransformer: {CONFIG['embedding']['model_name']}")
            except Exception as e:
                logger.error(f"Failed to initialize SentenceTransformer: {str(e)}")
                self.similarity_model = None
            
            self.available_metrics = {}
            metric_names = ['Hallucination', 'ContextPrecision']
            for metric_name in metric_names:
                try:
                    metric_module = importlib.import_module('opik.evaluation.metrics')
                    metric_class = getattr(metric_module, metric_name, None)
                    if metric_class:
                        self.available_metrics[metric_name.lower()] = metric_class
                        logger.info(f"Metric {metric_name} is available")
                        metric_instance = metric_class()
                        if hasattr(metric_instance, 'score'):
                            score_sig = inspect.signature(metric_instance.score)
                            logger.info(f"{metric_name}.score signature: {score_sig}")
                    else:
                        logger.warning(f"Metric {metric_name} not found in opik.evaluation.metrics")
                except Exception as e:
                    logger.warning(f"Failed to load metric {metric_name}: {str(e)}")
                    
            if not self.available_metrics:
                logger.error("No evaluation metrics available. Evaluation will return default scores.")
        except Exception as e:
            logger.error(f"Failed to initialize Opik client: {str(e)}")
            self.opik_client = None
            self.available_metrics = {}
            self.similarity_model = None

    def get_top_chunks(self, query: str, sds_paths: List[str]) -> List[str]:
        """
        Retrieves and reranks the top relevant document chunks for a given query.

        Args:
            query (str): The user's input query.
            sds_paths (List[str]): List of document paths to be used for retrieval.

        Returns:
            List[str]: A list of top-ranked chunk texts relevant to the query. Returns an empty list on failure.
        """

        try:
            rag_system = RAGSystem(sds_paths=sds_paths)
            retriever = rag_system.vectorstore.as_retriever(search_kwargs={"k": CONFIG["retriever"]["search_k"]})
            docs = retriever.invoke(query)
            expected_sources = [os.path.basename(path) for path in sds_paths]
            filtered_docs = [doc for doc in docs if doc.metadata.get("source") in expected_sources]
            
            chunk_texts = [doc.page_content for doc in filtered_docs]
            from app.reranker import rerank_chunks
            reranked_chunks = rerank_chunks(chunk_texts, query, k=len(chunk_texts))
            logger.info(f"Retrieved {len(reranked_chunks)} chunks for ground truth")
            return reranked_chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return []

    def evaluate_response(self, query: str, answer: str, ground_truth: List[str], is_web_search: bool = False, sds_paths: List[str] = []) -> Dict[str, float]:
        """
        Evaluates a generated answer against ground truth using hallucination and context precision metrics.

        Args:
            query (str): The original user query.
            answer (str): The generated response to be evaluated.
            ground_truth (List[str]): A list of reference chunks to evaluate against.
            is_web_search (bool, optional): Flag indicating if the response was generated via web search. Defaults to False.
            sds_paths (List[str], optional): Paths to documents used for ground truth retrieval if not provided. Defaults to [].

        Returns:
            Dict[str, float]: A dictionary containing evaluation scores for hallucination and context precision.
        """

        metrics = {
            "hallucination": 0.0,
            "context_precision": 0.0
        }
        
        if not self.opik_client or not self.available_metrics:
            logger.warning("Opik client or metrics not initialized. Returning default scores.")
            return metrics

        if not ground_truth and not is_web_search:
            ground_truth = self.get_top_chunks(query, sds_paths)
            if not ground_truth:
                logger.warning("No ground truth chunks retrieved. Returning default scores.")
                return metrics

        try:
            for metric_name, metric_class in self.available_metrics.items():
                try:
                    metric_instance = metric_class(model=f"groq/{self.model_name}")
                    if hasattr(metric_instance, 'score'):
                        if metric_name == "hallucination":
                            logger.info(f"Evaluating hallucination for query: {query[:50]}...")
                            score_result = metric_instance.score(query, answer, ground_truth)
                            score = score_result.score if score_result and hasattr(score_result, 'score') else 0.0
                            metrics["hallucination"] = float(score)
                            logger.info(f"Hallucination Score: {metrics['hallucination']:.4f}")
                        elif metric_name == "contextprecision":
                            logger.info(f"Evaluating context precision for query: {query[:50]}...")
                            is_based_on_chunk = False
                            if self.similarity_model:
                                answer_embedding = self.similarity_model.encode(answer, convert_to_tensor=True)
                                for chunk in ground_truth:
                                    chunk_embedding = self.similarity_model.encode(chunk, convert_to_tensor=True)
                                    similarity = util.cos_sim(answer_embedding, chunk_embedding).item()
                                    logger.info(f"Similarity score with chunk '{chunk[:50]}...': {similarity:.4f}")
                                    if similarity > 0.5:
                                        is_based_on_chunk = True
                                        break
                                if not is_based_on_chunk:
                                    answer_words = set(answer.lower().split())
                                    for chunk in ground_truth:
                                        chunk_words = set(chunk.lower().split())
                                        overlap = len(answer_words & chunk_words) / len(answer_words)
                                        logger.info(f"Keyword overlap with chunk '{chunk[:50]}...': {overlap:.4f}")
                                        if overlap > 0.5:
                                            is_based_on_chunk = True
                                            break
                            score = 1.0 if is_based_on_chunk else 0.0
                            metrics["context_precision"] = float(score)
                            logger.info(f"Context Precision custom score: {score:.4f}")
                        else:
                            logger.warning(f"Unknown metric {metric_name}. Skipping.")
                    else:
                        logger.warning(f"Metric {metric_name} has no 'score' method. Skipping.")
                except Exception as e:
                    logger.error(f"Error evaluating {metric_name}: {str(e)}", exc_info=True)
                    metrics["context_precision"] = 0.0

            self._save_evaluation_results(query, answer, ground_truth, metrics, is_web_search)
            logger.info(f"Final evaluation metrics: {metrics}")

            return metrics
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
            return metrics

    def _save_evaluation_results(self, query: str, answer: str, ground_truth: List[str], metrics: Dict[str, float], is_web_search: bool = False):
        """
        Saves evaluation results, including query, answer, metrics, and ground truth, to a JSON file.

        Args:
            query (str): The original user query.
            answer (str): The generated answer being evaluated.
            ground_truth (List[str]): Reference chunks used for evaluation.
            metrics (Dict[str, float]): Evaluation scores (e.g., hallucination, context precision).
            is_web_search (bool, optional): Indicates whether the query was answered using web search. Defaults to False.
        """

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"evaluation_{'web' if is_web_search else 'doc'}_{timestamp}.json")
            
            evaluation_data = {
                "query": query,
                "answer": answer,
                "ground_truth": ground_truth,
                "metrics": metrics,
                "timestamp": timestamp,
                "source": "web" if is_web_search else "document"
            }
            
            with open(output_file, "w") as f:
                json.dump(evaluation_data, f, indent=2)
            logger.info(f"Saved evaluation results to {output_file}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")