"""Central configuration. Every value can be overridden in a .env file (see .env.example)."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

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

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "")  # "" = auto (cuda if available, else cpu)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

# ChromaDB
COLLECTION_NAME = "taiwan_transport_laws"

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
# Minimum cosine similarity (0~1). Measured with bge-m3: relevant questions score
# ~0.65-0.75, off-topic questions ~0.35-0.45, so 0.5 separates them.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.5"))

# Ollama (local LLM)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# qwen2.5 handles Traditional Chinese well. 3b runs on CPU-only laptops;
# use qwen2.5:7b (or larger) for better answers if you have the RAM / GPU.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))
# Layers offloaded to GPU. Empty = let Ollama decide. Set 0 to force CPU, e.g. when an
# old / integrated GPU crashes the Vulkan backend ("does not support 16-bit storage").
OLLAMA_NUM_GPU = int(v) if (v := os.getenv("OLLAMA_NUM_GPU", "").strip()) else None
