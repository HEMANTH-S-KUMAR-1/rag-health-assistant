"""
embed_store.py
---------------
Step 2 of the RAG pipeline: compute an embedding vector per chunk and
persist an index to disk.

Embedding model: TF-IDF (scikit-learn TfidfVectorizer), word 1-2 grams.
Why TF-IDF and not a neural embedding model (e.g. sentence-transformers /
OpenAI embeddings): this environment has no network access to Hugging
Face or embedding APIs, so a neural model can't be downloaded or called
here. TF-IDF is a legitimate, fully offline, deterministic embedding
representation and is adequate for keyword-rich government-scheme text
(lots of named programmes like "AB-PMJAY", "PM-ABHIM" that TF-IDF weights
well). See IMPLEMENTATION_NOTE.md for the swap-in instructions to use
sentence-transformers ("all-MiniLM-L6-v2") instead - it's a ~5 line change
isolated entirely to this file, since retrieve.py only depends on
`vectorize(texts) -> matrix` and `vectorize_query(text) -> vector`.

Usage:
    python src/embed_store.py
    -> reads data/chunks.json
    -> writes index/vectorizer.joblib, index/matrix.joblib
"""
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks.json"
INDEX_DIR = Path(__file__).parent.parent / "index"
VECTORIZER_PATH = INDEX_DIR / "vectorizer.joblib"
MATRIX_PATH = INDEX_DIR / "matrix.joblib"


def build_index():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams capture scheme names like "jan aarogya"
        stop_words="english",
        sublinear_tf=True,    # dampens the effect of very frequent terms
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)  # shape: (n_chunks, n_features)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(matrix, MATRIX_PATH)

    print(f"Indexed {len(chunks)} chunks -> {matrix.shape[1]} TF-IDF features")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved matrix to {MATRIX_PATH}")


if __name__ == "__main__":
    build_index()
