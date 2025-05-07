from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END #type: ignore
from langchain_core.messages import AIMessage
import logging
from app.rag import RAGSystem
from app.web_search import WebSearchAgent
from app.config import CONFIG
from app.evaluation import EvaluationSystem

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    query: str
    doc_answer: dict
    confidence: float
    web_answer: dict
    final_answer: dict
    sds_paths: list
    evaluation_metrics: dict
    evaluate_metrics: bool

def doc_retrieval_node(state: AgentState) -> AgentState:
    """
    Document Retrieval Node function for processing the query and retrieving answers from a document retrieval system.

    Args:
        state (AgentState): The current state of the agent containing the query, paths for document retrieval, and other relevant data.

    Returns:
        AgentState: A new state object with the retrieved document answer, confidence score, and other relevant data.

    Raises:
        Exception: If an error occurs during the retrieval or response processing.

    Notes:
        - This function interacts with the RAG system to process the query and obtain an answer.
        - The response can either be a tuple of (answer, confidence) or a dictionary with answer, confidence, and metadata.
        - The function also keeps track of the confidence level for the retrieved answer.
    """

    logger.info(f"Document Retrieval Node: Processing query: {state['query']}")
    rag_system = RAGSystem(sds_paths=state["sds_paths"])
    response = rag_system.query(state["query"])
    if isinstance(response, tuple):
        answer, confidence = response
        doc_answer = {
            "answer": answer,
            "source": "document",
            "metadata": {}
        }
    else:
        doc_answer = {
            "answer": response["answer"],
            "source": "document",
            "metadata": response.get("metadata", {})
        }
        confidence = response.get("confidence", 0.0)
    logger.info(f"Document Retrieval Node: Confidence: {confidence:.2f}")
    return {
        "doc_answer": doc_answer,
        "confidence": confidence,
        "web_answer": state.get("web_answer", {}),
        "final_answer": doc_answer,
        "evaluation_metrics": state.get("evaluation_metrics", {}),
        "evaluate_metrics": state["evaluate_metrics"]
    }

def web_search_node(state: AgentState) -> AgentState:
    """
    Web Search Node function for processing a query and retrieving answers from a web search agent.

    Args:
        state (AgentState): The current state of the agent containing the query and other relevant data.

    Returns:
        AgentState: A new state object with the web search answer, ground truth, sources, and other relevant data.

    Raises:
        Exception: If an error occurs during the web search or response processing.

    Notes:
        - This function interacts with the WebSearchAgent to perform a web search and retrieve an answer.
        - The response includes the answer, metadata, ground truth, and sources.
        - The function maintains the retrieved web answer as well as the evaluation metrics from the previous state.
    """

    logger.info(f"Web Search Node: Processing query: {state['query']}")
    web_agent = WebSearchAgent()
    response = web_agent.search_web(state["query"])
    web_answer = {
        "answer": response["answer"],
        "source": "web",
        "metadata": response.get("metadata", {}),
        "ground_truth": response.get("ground_truth", []),
        "sources": response.get("sources", [])
    }
    return {
        "web_answer": web_answer,
        "final_answer": web_answer,
        "evaluation_metrics": state.get("evaluation_metrics", {}),
        "evaluate_metrics": state["evaluate_metrics"]
    }

def evaluation_node(state: AgentState) -> AgentState:
    """
    Evaluation Node function for evaluating the response to a query, based on the provided answer and ground truth.

    Args:
        state (AgentState): The current state of the agent containing the query, answer, ground truth, and other relevant data.

    Returns:
        AgentState: A new state object with the evaluation metrics, based on the evaluation of the query's response.

    Raises:
        Exception: If an error occurs during the evaluation process.

    Notes:
        - This function interacts with the `EvaluationSystem` to evaluate the generated response.
        - The evaluation checks the answer's accuracy using ground truth (if available) and returns relevant evaluation metrics.
        - The function also tracks whether the answer is sourced from a web search or a document.
        - The evaluation metrics are returned along with the existing `evaluate_metrics` from the previous state.
    """

    logger.info(f"Evaluation Node: Evaluating response for query: {state['query']}")
    evaluator = EvaluationSystem()
    ground_truth = state["final_answer"].get("ground_truth", []) if state["final_answer"]["source"] == "web" else []
    is_web_search = state["final_answer"]["source"] == "web"
    metrics = evaluator.evaluate_response(
        query=state["query"],
        answer=state["final_answer"]["answer"],
        ground_truth=ground_truth,
        is_web_search=is_web_search,
        sds_paths=state["sds_paths"]
    )
    return {
        "evaluation_metrics": metrics,
        "evaluate_metrics": state["evaluate_metrics"]
    }

def end_node(state: AgentState) -> AgentState:
    """
    End Node function to generate the final response, including the answer, evaluation metrics, and related metadata.

    Args:
        state (AgentState): The current state of the agent containing the final answer, evaluation metrics, and other relevant data.

    Returns:
        AgentState: A new state object with the final answer, evaluation metrics, and other relevant information.
    """

    logger.info("End Node: Final response")
    final_answer = {
        "answer": state["final_answer"]["answer"],
        "source": state["final_answer"]["source"],
        "metadata": state["final_answer"].get("metadata", {}),
        "ground_truth": state["final_answer"].get("ground_truth", []),
        "metrics": state["evaluation_metrics"],
        "sources": state["final_answer"].get("sources", [])
    }
    return {
        "final_answer": final_answer,
        "evaluation_metrics": state["evaluation_metrics"],
        "evaluate_metrics": state["evaluate_metrics"]
    }

def coordinator_node(state: AgentState) -> str:
    """
    Coordinator Node function to decide the next step in the agent's workflow based on confidence and evaluation metrics.

    Args:
        state (AgentState): The current state of the agent, including the confidence score and evaluation metrics.

    Returns:
        str: The name of the next node to execute, either "evaluation_node", "end_node", or "web_search" based on the decision logic.
    """

    confidence_threshold = CONFIG["agent"]["confidence_threshold"]
    logger.info(f"Coordinator Node: Confidence: {state['confidence']:.2f}, Threshold: {confidence_threshold}, Evaluate Metrics: {state['evaluate_metrics']}")
    if state["confidence"] >= confidence_threshold:
        logger.info("Coordinator Node: Using Document Agent response")
        return "evaluation_node" if state["evaluate_metrics"] else "end_node"
    logger.info("Coordinator Node: Triggering Web Search")
    return "web_search"

def build_graph():
    """
    Builds the agent's workflow graph, which defines the sequence of processing steps and conditions for transitions between nodes.
    The graph starts with document retrieval and uses a coordinator node to determine whether to proceed with a web search or directly evaluate the results. 
    The flow also includes an evaluation node and an end node.

    Returns:
        StateGraph: The compiled state graph defining the workflow of the agent.
    """

    workflow = StateGraph(AgentState)

    workflow.add_node("doc_retrieval", doc_retrieval_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("evaluation_node", evaluation_node)
    workflow.add_node("end_node", end_node)

    workflow.set_entry_point("doc_retrieval")
    workflow.add_conditional_edges(
        "doc_retrieval",
        coordinator_node,
        {
            "web_search": "web_search",
            "evaluation_node": "evaluation_node",
            "end_node": "end_node"
        }
    )
    workflow.add_edge("web_search", "evaluation_node" if CONFIG.get("evaluate_metrics", True) else "end_node")
    workflow.add_edge("evaluation_node", "end_node")
    workflow.add_edge("end_node", END)

    return workflow.compile()

def run_agent_workflow(query: str, sds_paths: list, evaluate_metrics: bool = False) -> dict:
    """
    Runs the agent's workflow, processing a given query through the defined stages of document retrieval, web search (if necessary), evaluation, and final response.

    The function initiates the agent with an initial state, invokes the workflow, and returns the final answer after the processing flow is completed.

    Args:
        query (str): The natural language query that the agent will process.
        sds_paths (list): A list of paths to the source data structures (e.g., documents, databases) used for retrieval.
        evaluate_metrics (bool, optional): Flag indicating whether to evaluate the response metrics (default is False).

    Returns:
        dict: A dictionary containing the final answer, metadata, and evaluation metrics.
    """

    graph = build_graph()
    initial_state = {
        "query": query,
        "doc_answer": {},
        "confidence": 0.0,
        "web_answer": {},
        "final_answer": {},
        "sds_paths": sds_paths,
        "evaluation_metrics": {},
        "evaluate_metrics": evaluate_metrics
    }
    result = graph.invoke(initial_state)
    final_answer = result["final_answer"]
    logger.info(f"Final answer: {final_answer['answer'][:100]}...")
    return final_answer
