from pathlib import Path

# ── Root of the project ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ─────────────────────────────────────────────────────────────
RAW_PDFS_DIR     = BASE_DIR / "data" / "raw_pdfs"
PROCESSED_DIR    = BASE_DIR / "data" / "processed"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# ── Embedding model (HuggingFace) ───────────────────────────────────────────
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"

# ── Text splitting settings ─────────────────────────────────────────────────
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 100