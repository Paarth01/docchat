"""PDF ingestion: load, chunk, embed, persist to ChromaDB."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _collect_pdf_paths(pdf_folder: str) -> list[Path]:
    root = Path(pdf_folder).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PDF folder does not exist or is not a directory: {root}")
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def ingest_pdfs(pdf_folder: str) -> dict[str, int]:
    """
    Recursively scan pdf_folder for PDFs, chunk, embed locally, and upsert into Chroma.
    If a persisted collection already exists, new chunks are added (not replaced).
    """
    pdf_paths = _collect_pdf_paths(pdf_folder)
    if not pdf_paths:
        logger.warning("No PDF files found under %s", pdf_folder)

    all_docs: list[Any] = []
    for pdf_path in pdf_paths:
        logger.info("Loading PDF: %s", pdf_path)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for doc in pages:
            meta = dict(doc.metadata) if doc.metadata else {}
            meta["source"] = pdf_path.name
            if "page" in meta:
                meta["page"] = int(meta["page"])
            else:
                meta["page"] = 0
            doc.metadata = meta
        all_docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)
    logger.info("Created %d chunks from %d raw documents", len(chunks), len(all_docs))

    embeddings = _get_embeddings()
    os.makedirs(CHROMA_PERSIST_PATH, exist_ok=True)
    persist_nonempty = Path(CHROMA_PERSIST_PATH).exists() and any(Path(CHROMA_PERSIST_PATH).iterdir())

    if persist_nonempty:
        store = Chroma(
            persist_directory=CHROMA_PERSIST_PATH,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
        )
        if chunks:
            store.add_documents(chunks)
            logger.info("Added %d documents to existing Chroma collection", len(chunks))
    elif chunks:
        store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_PATH,
            collection_name=CHROMA_COLLECTION_NAME,
        )
        logger.info("Created new Chroma collection with %d documents", len(chunks))
    else:
        logger.info("No PDF chunks to index; vector store not created or updated.")
        return {
            "files_processed": len(pdf_paths),
            "chunks_created": 0,
            "collection_size": 0,
        }

    collection_size = int(store._collection.count())  # noqa: SLF001 — Chroma public surface is limited

    return {
        "files_processed": len(pdf_paths),
        "chunks_created": len(chunks),
        "collection_size": collection_size,
    }


def get_vectorstore() -> Chroma:
    """Load persisted Chroma. Raises a clear error if the store has not been created yet."""
    path = Path(CHROMA_PERSIST_PATH)
    if not path.exists() or not any(path.iterdir()):
        raise FileNotFoundError(
            f"No vector database found at '{CHROMA_PERSIST_PATH}'. "
            "Run POST /ingest after placing PDFs in the configured PDF folder."
        )

    embeddings = _get_embeddings()
    try:
        return Chroma(
            persist_directory=CHROMA_PERSIST_PATH,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Chroma vector store from '{CHROMA_PERSIST_PATH}': {exc}"
        ) from exc
