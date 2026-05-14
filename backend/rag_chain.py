"""LangChain retrieval + stuff-documents chain (non-graph pipeline)."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL, TOP_K_RESULTS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant. Answer the question using ONLY the context provided below. If the answer is not in the context, say 'I could not find an answer in the provided documents.' Always end your response by listing the source filenames you used under a 'Sources:' heading.

Context:
{context}

Question: {input}"""


def _unique_sources_from_docs(docs: list[Any]) -> list[str]:
    names: list[str] = []
    for d in docs or []:
        src = getattr(d, "metadata", None) or {}
        if isinstance(src, dict):
            name = src.get("source")
            if name and name not in names:
                names.append(str(name))
    return names


def _parse_sources_from_answer(text: str) -> list[str]:
    if not text:
        return []
    m = re.search(r"Sources:\s*(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    block = m.group(1).strip()
    lines = [ln.strip(" -*\t") for ln in block.splitlines() if ln.strip()]
    out: list[str] = []
    for ln in lines:
        if ln and ln not in out:
            out.append(ln)
    return out


def build_rag_chain(vectorstore: Chroma) -> Runnable:
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", SYSTEM_PROMPT),
        ]
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    logger.info("Built LangChain retrieval chain (model=%s)", OLLAMA_MODEL)
    return chain


def query_chain(chain: Runnable, question: str) -> dict[str, Any]:
    """Run the retrieval chain and normalize the response shape."""
    result = chain.invoke({"input": question})
    answer = result.get("answer")
    if hasattr(answer, "content"):
        answer = answer.content
    answer = str(answer or "")
    context = result.get("context") or []
    chunks_used = len(context) if isinstance(context, list) else 0
    sources = _unique_sources_from_docs(context) if chunks_used else _parse_sources_from_answer(answer)
    return {"answer": answer, "sources": sources, "chunks_used": chunks_used}
