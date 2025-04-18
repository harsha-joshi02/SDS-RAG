from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END # type: ignore
from langchain_core.messages import AIMessage
import logging
from app.rag import RAGSystem
from app.web_search import WebSearchAgent
from app.config import CONFIG

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    query: str
    doc_answer: str
    confidence: float
    web_answer: str
    final_answer: str
    sds_paths: list 

def doc_retrieval_node(state: AgentState) -> AgentState:
    logger.info(f"Document Retrieval Node: Processing query: {state['query']}")
    rag_system = RAGSystem(sds_paths=state["sds_paths"])
    doc_answer, confidence = rag_system.query(state["query"])
    logger.info(f"Document Retrieval Node: Confidence: {confidence:.2f}")
    return {
        "doc_answer": doc_answer,
        "confidence": confidence,
        "web_answer": state.get("web_answer", ""),
        "final_answer": doc_answer  
    }

def web_search_node(state: AgentState) -> AgentState:
    logger.info(f"Web Search Node: Processing query: {state['query']}")
    web_agent = WebSearchAgent()
    web_answer = web_agent.search_web(state["query"])
    return {
        "web_answer": web_answer,
        "final_answer": web_answer 
    }

def end_node(state: AgentState) -> AgentState:
    logger.info("End Node: Final response")
    return {
        "final_answer": state["final_answer"] or state["doc_answer"]
    }

def coordinator_node(state: AgentState) -> str:
    confidence_threshold = CONFIG["agent"]["confidence_threshold"]
    logger.info(f"Coordinator Node: Confidence: {state['confidence']:.2f}, Threshold: {confidence_threshold}")
    if state["confidence"] >= confidence_threshold:
        logger.info("Coordinator Node: Using Document Agent response")
        return "end_node"
    logger.info("Coordinator Node: Triggering Web Search")
    return "web_search"

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("doc_retrieval", doc_retrieval_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("end_node", end_node)

    workflow.set_entry_point("doc_retrieval")
    workflow.add_conditional_edges(
        "doc_retrieval",
        coordinator_node,
        {
            "web_search": "web_search",
            "end_node": "end_node"
        }
    )
    workflow.add_edge("web_search", "end_node")
    workflow.add_edge("end_node", END) 

    return workflow.compile()

def run_agent_workflow(query: str, sds_paths: list) -> str:
    graph = build_graph()
    initial_state = {
        "query": query,
        "doc_answer": "",
        "confidence": 0.0,
        "web_answer": "",
        "final_answer": "",
        "sds_paths": sds_paths  
    }
    result = graph.invoke(initial_state)
    final_answer = result["final_answer"]
    logger.info(f"Final answer: {final_answer[:100]}...")
    return final_answer