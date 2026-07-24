"""
retrieve.py
-----------
Step 3 of the RAG pipeline: given a user question, embed it with the same
model used for the chunks, then return the top-k most similar chunks
by cosine similarity.

Features:
- Semantic search via sentence-transformers embeddings
- Optional LLM-based re-ranking (--rerank flag in CLI)
- Multi-hop detection for comparison/multi-entity questions
"""
import json
import os
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks.json"
INDEX_DIR = Path(__file__).parent.parent / "index"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npz"


class Retriever:
    def __init__(self):
        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                "Index not found. Run `python src/embed_store.py` first."
            )
        self.chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
        self.matrix = data["embeddings"]
        model_name = str(data["model_name"])
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = 3):
        """
        Returns a list of {chunk, score} dicts, sorted by descending
        cosine similarity to the query, length top_k.
        """
        query_vec = self.model.encode([query], normalize_embeddings=True)
        scores = cosine_similarity(query_vec, self.matrix)[0]  # shape: (n_chunks,)

        ranked_indices = scores.argsort()[::-1][:top_k]
        results = [
            {"chunk": self.chunks[i], "score": float(scores[i])}
            for i in ranked_indices
        ]
        return results

    def search_multihop(self, query: str, top_k: int = 3):
        """
        Detect multi-entity questions (e.g. 'compare AB-PMJAY and PM-ABHIM')
        and retrieve relevant chunks for each entity separately, then combine.
        Falls back to standard search for single-entity questions.
        """
        # Detect multi-hop patterns
        multihop_patterns = [
            r"\bcompare\b", r"\bvs\.?\b", r"\bbetween\b",
            r"\band\b.*\band\b",  # multiple "and"s
        ]
        is_multihop = any(re.search(p, query, re.IGNORECASE) for p in multihop_patterns)

        if not is_multihop:
            return self.search(query, top_k=top_k)

        # Extract sub-queries by splitting on conjunctions
        sub_queries = re.split(r"\band\b|\bvs\.?\b|\bcompare\b|\bbetween\b", query, flags=re.IGNORECASE)
        sub_queries = [sq.strip() for sq in sub_queries if sq.strip() and len(sq.strip()) > 3]

        if len(sub_queries) < 2:
            return self.search(query, top_k=top_k)

        # Retrieve top-1 for each sub-query, then fill remaining slots
        seen_ids = set()
        results = []

        for sq in sub_queries:
            for r in self.search(sq, top_k=1):
                cid = r["chunk"]["id"]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    results.append(r)

        # Fill remaining with overall query results
        if len(results) < top_k:
            for r in self.search(query, top_k=top_k + len(sub_queries)):
                cid = r["chunk"]["id"]
                if cid not in seen_ids and len(results) < top_k:
                    seen_ids.add(cid)
                    results.append(r)

        return results[:top_k]

    def search_with_rerank(self, query: str, top_k: int = 3, rerank_pool: int = 8):
        """
        Two-stage retrieval: retrieve top-N by embeddings, then re-rank
        with Gemini by asking it to score relevance of each chunk.
        Falls back to plain search if no API key is set.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return self.search(query, top_k=top_k)

        # Stage 1: broad retrieval
        candidates = self.search(query, top_k=rerank_pool)

        # Stage 2: LLM re-ranking
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=api_key)
            model_name = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

            rerank_prompt = (
                f"Question: {query}\n\n"
                "Rate the relevance of each text passage below to the question. "
                "Return ONLY a JSON array of scores from 0 to 10, one per passage, "
                "in the same order.\n\n"
            )
            for i, r in enumerate(candidates):
                snippet = r["chunk"]["text"][:300]
                rerank_prompt += f"Passage {i+1}: {snippet}\n\n"

            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=100,
                    temperature=0.0,
                )
            )
            response = model.generate_content(rerank_prompt)

            # Parse scores from response
            response_text = response.text
            scores_match = re.search(r"\[[\d,\s\.]+\]", response_text)
            if scores_match:
                import json as json_mod
                rerank_scores = json_mod.loads(scores_match.group())
                for i, score in enumerate(rerank_scores[:len(candidates)]):
                    candidates[i]["rerank_score"] = float(score)
                candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        except Exception:
            # Fall back to embedding-only ranking on any error
            pass

        return candidates[:top_k]


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "What is AB-PMJAY?"
    retriever = Retriever()
    for r in retriever.search(query, top_k=3):
        print(f"[{r['score']:.3f}] {r['chunk']['title']}")
