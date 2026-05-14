# DocChat

**DocChat** is a small, production-minded local RAG chatbot: ask questions over your own PDFs, fully offline, with **Ollama** as the LLM and **ChromaDB** + **sentence-transformers** for retrieval.

## Prerequisites

- **Python 3.11+** (required — DocChat is not tested on older interpreters)
- **Ollama** installed and running locally
- The **mistral** model pulled: `ollama pull mistral`

Before creating the venv, confirm the interpreter: `python --version` should report **3.11.x or newer**. On Windows, if `python` still points at an older release, use the **Python 3.11+** launcher instead, for example `py -3.11 -m venv .venv`, so `pip install` pulls pre-built wheels and you avoid a from-source `greenlet` build that requires **Microsoft C++ Build Tools**.

## Setup

1. Open a terminal in the `docchat` directory (the folder that contains `backend/`, `frontend/`, and `requirements.txt`).
2. Create a virtual environment (recommended) and install dependencies:
  ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
  ```
   On Windows, if `python` is not 3.11+, prefer: `py -3.11 -m venv .venv` then activate and `pip install` as above.
3. (Optional) Copy `.env.example` to `.env` if you want to document local overrides for your environment.
4. Start the API (from the `docchat` directory):
  ```bash
   python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
  ```
5. Open the UI in a browser: `http://127.0.0.1:8000/`

## Usage

1. Copy PDFs into `data/pdfs/` (subfolders are fine; ingestion is recursive).
2. Click **Ingest PDFs** in the sidebar (or `POST /ingest`).
3. Choose **Langgraph** or **Langchain** as the pipeline, then chat.

The vector database is written to `chroma_db/` on disk and reused across restarts.

## API


| Method | Path      | Description                                                                                                                                                                                                                                                                                 |
| ------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/health` | Liveness probe: `{ "status": "ok" }`.                                                                                                                                                                                                                                                       |
| `GET`  | `/status` | Ollama reachability (`/api/tags`), indexed chunk count, active LLM and embedding model names.                                                                                                                                                                                               |
| `POST` | `/ingest` | Body: `{ "folder_path": "<optional path>" }`. Defaults to configured `data/pdfs`. Loads PDFs, chunks, embeds locally, persists to Chroma. Returns `status`, `files_processed`, `chunks_created`.                                                                                            |
| `POST` | `/query`  | Body: `{ "question": "<text>", "use_graph": true }`. If `use_graph` is `true`, runs the **LangGraph** `StateGraph` pipeline; if `false`, runs the **LangChain** retrieval + stuff-documents chain. Returns `answer`, `sources`, `chunks_used`, and `pipeline` (`langgraph` or `langchain`). |


Errors return JSON with a `detail` field (FastAPI default). If nothing has been ingested yet, `/query` responds with **400** and: `No documents found. Please ingest PDFs first via POST /ingest`.

## Architecture

DocChat ships **two interchangeable RAG paths** over the same Chroma vector store:

1. **LangGraph (`use_graph: true`)** — A `StateGraph` with two nodes: **retrieve** (`similarity_search` with `TOP_K`) and **generate** (prompt + **ChatOllama**). This demonstrates explicit graph orchestration, easy extension (e.g. re-ranking, tool calls, branching), and clear state (`question`, `retrieved_docs`, `answer`, `sources`).
2. **LangChain chain (`use_graph: false`)** — `create_retrieval_chain` + `create_stuff_documents_chain` with the same retrieval width and system instructions. This is the classic concise RAG stack for comparison and interoperability.

Both use the same embeddings, chunking policy, and Ollama model configured in `backend/config.py`.

## Project structure

```
docchat/
├── backend/
│   ├── __init__.py
│   ├── main.py        # FastAPI app, CORS, routes, lifespan
│   ├── ingest.py      # PDF load → split → embed → Chroma
│   ├── rag_chain.py   # LangChain retrieval + stuff chain
│   ├── graph.py       # LangGraph StateGraph RAG
│   └── config.py      # Paths, models, chunking, top-k
├── frontend/
│   └── index.html     # Single-file dark UI
├── data/
│   └── pdfs/          # Drop PDFs here
├── chroma_db/         # Created by Chroma after first ingest
├── requirements.txt
├── .env.example
└── README.md
```

