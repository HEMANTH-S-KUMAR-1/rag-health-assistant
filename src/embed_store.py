"""
embed_store.py
---------------
Step 2 of the RAG pipeline: compute an embedding vector per chunk and
persist an index to disk.

Embedding model: sentence-transformers/all-mpnet-base-v2 (768-dim).
Why all-mpnet-base-v2 and not all-MiniLM-L6-v2: mpnet has higher
semantic quality (scores ~63 on STS benchmark vs ~59 for MiniLM), and
at 19 chunks the extra compute is negligible.

Storage: NumPy .npz compressed arrays on local disk. At 19 chunks,
a dedicated vector database (FAISS, Chroma, Pinecone) is overhead.

Usage:
    python src/embed_store.py
    -> reads data/chunks.json
    -> writes index/embeddings.npz
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks.json"
INDEX_DIR = Path(__file__).parent.parent / "index"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npz"
MODEL_NAME = "all-mpnet-base-v2"


def build_index():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]

    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Encoding {len(chunks)} chunks...")
    matrix = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMBEDDINGS_PATH, embeddings=matrix, model_name=MODEL_NAME)

    print(f"Indexed {len(chunks)} chunks -> {matrix.shape[1]}-dim embeddings")
    print(f"Saved to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    build_index()
