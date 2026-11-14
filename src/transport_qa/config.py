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
