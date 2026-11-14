"""Central configuration: paths and chunking parameters."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

# Chunking (unit: Chinese characters).
# Most articles are short, so one article = one chunk. Only articles longer
# than CHUNK_SIZE are split further, with CHUNK_OVERLAP to preserve context.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# Embedding model (multilingual, strong on Traditional Chinese; 1024-dim dense vectors).
# Override with environment variables if needed, e.g. EMBEDDING_DEVICE=cuda
import os  # noqa: E402

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "")  # "" = auto (cuda if available, else cpu)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

# ChromaDB
COLLECTION_NAME = "taiwan_transport_laws"
