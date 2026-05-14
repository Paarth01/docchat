"""FastAPI server for DocChat — ingest, query, status, health."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()

from .config import (
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    PDF_FOLDER,
)
from .graph import build_rag_graph, run_graph
from .ingest import get_vectorstore, ingest_pdfs
from .rag_chain import build_rag_chain, query_chain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("docchat")

_vectorstore: Any = None
_rag_chain: Any = None
_rag_graph: Any = None


def _reload_pipeline() -> None:
    global _vectorstore, _rag_chain, _rag_graph
    logger.info("Loading vector store and rebuilding RAG pipelines…")
    _vectorstore = get_vectorstore()
    _rag_chain = build_rag_chain(_vectorstore)
    _rag_graph = build_rag_graph(_vectorstore)
    logger.info("Pipelines ready.")


def _store_has_documents() -> bool:
    if _vectorstore is None:
        return False
    try:
        sample = _vectorstore.get(limit=1)
        ids = sample.get("ids") if isinstance(sample, dict) else None
        return bool(ids)
    except Exception:
        logger.exception("Failed to probe vector store for documents")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _reload_pipeline()
    except FileNotFoundError as exc:
        logger.warning("Vector store not ready at startup: %s", exc)
    except Exception:
        logger.exception("Startup pipeline load failed — ingest will be required")
    yield


app = FastAPI(title="DocChat", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


@app.get("/")
def serve_ui() -> FileResponse:
    index = _FRONTEND_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index)


class IngestBody(BaseModel):
    folder_path: str | None = Field(
        default=None,
        description="Optional folder containing PDFs (recursive). Defaults to configured PDF_FOLDER.",
    )


class QueryBody(BaseModel):
    question: str = Field(..., min_length=1)
    use_graph: bool = True


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
async def status() -> dict[str, Any]:
    ollama_connected = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags")
            ollama_connected = r.status_code == 200
    except Exception:
        logger.debug("Ollama not reachable at %s", OLLAMA_BASE_URL, exc_info=True)

    documents_indexed = 0
    if _vectorstore is not None:
        try:
            documents_indexed = int(_vectorstore._collection.count())  # noqa: SLF001
        except Exception:
            logger.exception("Could not read Chroma document count")

    return {
        "ollama_connected": ollama_connected,
        "documents_indexed": documents_indexed,
        "model": OLLAMA_MODEL,
        "embedding_model": EMBEDDING_MODEL,
    }


@app.post("/ingest")
async def ingest(body: IngestBody | None = Body(default=None)) -> dict[str, Any]:
    folder = (body.folder_path if body and body.folder_path else None) or PDF_FOLDER
    logger.info("Ingest requested for folder: %s", folder)
    try:
        summary = ingest_pdfs(folder)
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        _reload_pipeline()
    except FileNotFoundError:
        if summary["chunks_created"] > 0:
            logger.exception("Reload after ingest failed: vector store missing despite new chunks")
            raise HTTPException(
                status_code=500,
                detail="Ingest reported new chunks but the vector store could not be loaded.",
            ) from None
        logger.warning("Ingest completed with no indexed chunks; vector store not initialized.")
    except Exception as exc:
        logger.exception("Reload after ingest failed")
        raise HTTPException(
            status_code=500,
            detail=f"Ingest succeeded but failed to reload vector store: {exc}",
        ) from exc

    return {
        "status": "success",
        "files_processed": summary["files_processed"],
        "chunks_created": summary["chunks_created"],
    }


@app.post("/query")
async def query(body: QueryBody) -> dict[str, Any]:
    if _vectorstore is None or not _store_has_documents():
        raise HTTPException(
            status_code=400,
            detail="No documents found. Please ingest PDFs first via POST /ingest",
        )

    use_graph = body.use_graph
    logger.info("Query (graph=%s): %r", use_graph, body.question[:200])

    try:
        if use_graph:
            if _rag_graph is None:
                raise HTTPException(status_code=503, detail="LangGraph pipeline is not initialized.")
            out = run_graph(_rag_graph, body.question)
            pipeline = "langgraph"
        else:
            if _rag_chain is None:
                raise HTTPException(status_code=503, detail="LangChain pipeline is not initialized.")
            out = query_chain(_rag_chain, body.question)
            pipeline = "langchain"
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Query failed")
        msg = str(exc).lower()
        if "connection" in msg or "refused" in msg or "connect" in msg:
            raise HTTPException(
                status_code=503,
                detail="Ollama does not appear to be running or reachable. Start Ollama and ensure "
                f"'{OLLAMA_MODEL}' is available (e.g. `ollama pull mistral`).",
            ) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "answer": out["answer"],
        "sources": out["sources"],
        "chunks_used": out["chunks_used"],
        "pipeline": pipeline,
    }
