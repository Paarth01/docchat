"""Central configuration for DocChat — all tunables live here."""

from pathlib import Path

# Project root (parent of backend/)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

OLLAMA_MODEL = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PERSIST_PATH = str(_PROJECT_ROOT / "chroma_db")
PDF_FOLDER = str(_PROJECT_ROOT / "data" / "pdfs")
CHROMA_COLLECTION_NAME = "docchat"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 4
