"""
retrieve.py
-----------
Step 3 of the RAG pipeline: given a user question, embed it with the same
vectorizer used for the chunks, then return the top-k most similar chunks
by cosine similarity.
"""
import json
from pathlib import Path

import joblib
from sklearn.metrics.pairwise import cosine_similarity

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks.json"
INDEX_DIR = Path(__file__).parent.parent / "index"
VECTORIZER_PATH = INDEX_DIR / "vectorizer.joblib"
MATRIX_PATH = INDEX_DIR / "matrix.joblib"


class Retriever:
    def __init__(self):
        if not VECTORIZER_PATH.exists() or not MATRIX_PATH.exists():
            raise FileNotFoundError(
                "Index not found. Run `python src/embed_store.py` first."
            )
        self.chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.matrix = joblib.load(MATRIX_PATH)

    def search(self, query: str, top_k: int = 3):
        """
        Returns a list of {chunk, score} dicts, sorted by descending
        cosine similarity to the query, length top_k.
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]  # shape: (n_chunks,)

        ranked_indices = scores.argsort()[::-1][:top_k]
        results = [
            {"chunk": self.chunks[i], "score": float(scores[i])}
            for i in ranked_indices
        ]
        return results


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What is AB-PMJAY?"
    retriever = Retriever()
    for r in retriever.search(query, top_k=3):
        print(f"[{r['score']:.3f}] {r['chunk']['title']}")
