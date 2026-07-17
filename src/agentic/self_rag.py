"""
DocuMind Self-RAG (Corrective RAG)
-------------------------------------
v2 retrieval strategy: instead of a single retrieve-then-generate pass,
this loop checks whether retrieval was actually good enough before
generating an answer. If not, it reformulates the query and tries again,
up to a hard retry budget (to prevent infinite loops).

Confidence signal: the cross-encoder reranker's own top score. This is
deliberately NOT a separate LLM judge call — the reranker already scores
relevance as part of the existing pipeline, so reusing it costs zero
extra latency/tokens. A raw score above SELF_RAG_CONFIDENCE_THRESHOLD is
treated as "good enough to answer"; below that, the query gets rewritten
and retried.

Flow:
    retrieve → confident enough? ──yes──> generate → END
                     │no (retries left)
                     v
                reformulate query → retrieve (loop)
                     │no retries left
                     v
                generate anyway (existing prompt rule makes it refuse
                rather than hallucinate, since context is weak)

This is the v1 pipeline's exact retrieve/rerank/generate components,
just wrapped in a LangGraph state machine — v1 (pipeline.query) is left
completely untouched so it remains directly comparable via RAGAS.
"""
from __future__ import annotations

import uuid
from typing import Any, List, Optional, Tuple, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

REFORMULATE_SYSTEM_PROMPT = """You rewrite search queries for a document retrieval system.
The previous query did not retrieve confident, relevant results.
Rewrite it to be more specific, use alternate terminology, or break down
compound questions — whatever is most likely to retrieve better matches.
Return ONLY the rewritten query text, nothing else."""


class SelfRAGState(TypedDict):
    original_question: str
    current_query: str
    user_id: Optional[uuid.UUID]
    retry_count: int
    max_retries: int
    confidence_threshold: float
    candidates: List[Tuple[Document, float]]
    top_score: float
    reformulation_history: List[str]
    result: Optional[dict]


def build_self_rag_graph(pipeline):
    """
    Build the Self-RAG state graph, closing over the given RAGPipeline's
    already-initialised retriever/reranker/generator (no duplicate model
    loading).
    """
    settings = get_settings()

    def retrieve_node(state: SelfRAGState) -> dict:
        candidates = pipeline._retriever.retrieve(state["current_query"], user_id=state["user_id"])
        reranked = pipeline._reranker.rerank(state["current_query"], candidates)
        top_score = reranked[0][1] if reranked else float("-inf")
        logger.info(
            "Self-RAG retrieve",
            attempt=state["retry_count"],
            query=state["current_query"][:80],
            top_score=round(top_score, 3),
            num_chunks=len(reranked),
        )
        return {"candidates": reranked, "top_score": top_score}

    def route_after_retrieve(state: SelfRAGState) -> str:
        confident = state["top_score"] >= state["confidence_threshold"]
        retries_left = state["retry_count"] < state["max_retries"]
        if confident or not retries_left:
            return "generate"
        return "reformulate"

    def reformulate_node(state: SelfRAGState) -> dict:
        llm = pipeline._generator._get_llm()
        messages = [
            SystemMessage(content=REFORMULATE_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Original question: {state['original_question']}\n"
                f"Previous query tried: {state['current_query']}\n"
                f"Rewrite it:"
            ),
        ]
        new_query = llm.invoke(messages).content.strip()
        logger.info("Self-RAG reformulated query", attempt=state["retry_count"] + 1, new_query=new_query[:80])
        return {
            "current_query": new_query,
            "retry_count": state["retry_count"] + 1,
            "reformulation_history": state["reformulation_history"] + [new_query],
        }

    def generate_node(state: SelfRAGState) -> dict:
        result = pipeline._generator.generate(state["original_question"], state["candidates"])
        result["chunks_used"] = len(state["candidates"])
        result["self_rag_retries"] = state["retry_count"]
        result["self_rag_reformulations"] = state["reformulation_history"]
        result["self_rag_final_confidence"] = state["top_score"]
        result["contexts"] = [doc.page_content for doc, _score in state["candidates"]]
        return {"result": result}

    graph = StateGraph(SelfRAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve", route_after_retrieve, {"generate": "generate", "reformulate": "reformulate"}
    )
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


def run_self_rag(pipeline, question: str, user_id: Optional[uuid.UUID] = None) -> dict:
    """Entry point: run the Self-RAG loop for a single question."""
    settings = get_settings()
    graph = build_self_rag_graph(pipeline)

    initial_state: SelfRAGState = {
        "original_question": question,
        "current_query": question,
        "user_id": user_id,
        "retry_count": 0,
        "max_retries": settings.self_rag.self_rag_max_retries,
        "confidence_threshold": settings.self_rag.self_rag_confidence_threshold,
        "candidates": [],
        "top_score": float("-inf"),
        "reformulation_history": [],
        "result": None,
    }

    final_state = graph.invoke(initial_state)
    return final_state["result"]