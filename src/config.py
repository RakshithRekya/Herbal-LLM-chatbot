from pathlib import Path

# ── Root of the project ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ─────────────────────────────────────────────────────────────
RAW_PDFS_DIR     = BASE_DIR / "data" / "raw_pdfs"
PROCESSED_DIR    = BASE_DIR / "data" / "processed"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# ── Ollama model settings ───────────────────────────────────────────────────
LLM_MODEL       = "qwen2.5:7b"
EMBEDDING_MODEL = "nomic-embed-text"

# ── Text splitting settings ─────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50