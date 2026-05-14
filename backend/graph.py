"""LangGraph StateGraph pipeline: retrieve → generate."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL, TOP_K_RESULTS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant. Answer the question using ONLY the context provided below. If the answer is not in the context, say 'I could not find an answer in the provided documents.' Always end your response by listing the source filenames you used under a 'Sources:' heading.

Context:
{context}

Question: {question}"""


class RAGState(TypedDict, total=False):
    question: str
    retrieved_docs: list
    answer: str
    sources: list[str]


def _format_context(docs: list[Any]) -> str:
    parts: list[str] = []
    for i, doc in enumerate(docs or [], start=1):
        text = getattr(doc, "page_content", "") or ""
        parts.append(f"[{i}] {text}")
    return "\n\n".join(parts)


def _sources_from_docs(docs: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for d in docs or []:
        meta = getattr(d, "metadata", None) or {}
        if isinstance(meta, dict):
            src = meta.get("source")
            if src and src not in seen:
                seen.add(str(src))
                out.append(str(src))
    return out


def build_rag_graph(vectorstore: Chroma) -> Any:
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    def retrieve_node(state: RAGState) -> dict[str, Any]:
        q = state.get("question") or ""
        docs = vectorstore.similarity_search(q, k=TOP_K_RESULTS)
        logger.debug("retrieve_node: %d docs for question=%r", len(docs), q[:80])
        return {"retrieved_docs": docs}

    def generate_node(state: RAGState) -> dict[str, Any]:
        q = state.get("question") or ""
        docs = state.get("retrieved_docs") or []
        context = _format_context(docs)
        sources = _sources_from_docs(docs)
        prompt_text = SYSTEM_PROMPT.format(context=context, question=q)
        messages = [HumanMessage(content=prompt_text)]
        try:
            resp = llm.invoke(messages)
        except Exception:
            logger.exception("ChatOllama invoke failed in generate_node")
            raise
        content = getattr(resp, "content", None) or str(resp)
        return {"answer": str(content), "sources": sources}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    compiled = graph.compile()
    logger.info("Compiled LangGraph RAG pipeline (model=%s)", OLLAMA_MODEL)
    return compiled


def run_graph(graph: Any, question: str) -> dict[str, Any]:
    final = graph.invoke({"question": question})
    docs = final.get("retrieved_docs") or []
    chunks_used = len(docs) if isinstance(docs, list) else 0
    return {
        "answer": str(final.get("answer") or ""),
        "sources": list(final.get("sources") or []),
        "chunks_used": chunks_used,
    }
