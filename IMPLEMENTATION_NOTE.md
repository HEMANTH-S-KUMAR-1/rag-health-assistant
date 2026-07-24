# Implementation Note — RAG Q&A Assistant (India's Health Transformation)

## 1. Choices

### Chunking strategy
Header-aligned splitting rather than a fixed-size sliding window. The
source document is already organised under named sections (AB-PMJAY,
Ayushman Arogya Mandirs, PM-ABHIM, ABDM, NHM sub-programmes), and the
assignment explicitly asks for chunks that map to those sections. A
sliding window would frequently cut a scheme's description in half. The
trade-off: section lengths in the source are uneven, so a rule-based
merge/split step was needed to keep every chunk within roughly 200–500
words while preserving topic boundaries (see `ingest.py` docstring for the
exact algorithm). Result: 11 chunks, 168–434 words, average 325.

### Embedding model: TF-IDF (not a neural embedding model)
This was a constraint-driven decision, not a preference. The development
environment used to build this assignment has no network path to Hugging
Face, OpenAI, or any embeddings API — only a small allowlist of package
registries (PyPI, npm, GitHub) is reachable. Under that constraint,
`scikit-learn`'s `TfidfVectorizer` (word 1–2 grams, English stop words
removed, sublinear TF scaling) is the strongest fully-offline, licence-free
option, and it is a legitimate embedding technique, not a placeholder.

It also happens to suit this particular document well: government-scheme
text is dense with distinctive proper nouns and acronyms (AB-PMJAY,
ABHA, PM-ABHIM, Tele-MANAS...), and TF-IDF's term-weighting rewards exactly
that kind of lexical distinctiveness. In manual testing, top-1 retrieval
correctly matched all spot-check questions (e.g. "senior citizens"
correctly surfaced the AB-PMJAY / Vay Vandana chunk with the top score
roughly 20x the next-best match).

**Limitation, stated plainly**: TF-IDF is a lexical/keyword method, not a
semantic one — it will not match a question that uses no vocabulary
overlap with the source (e.g. a question phrased entirely in synonyms).
A true embedding model (sentence-transformers `all-MiniLM-L6-v2`, or an
API-based embedding model) would generalise better to paraphrased
questions. The codebase isolates this choice to `embed_store.py` and
`retrieve.py`'s `vectorizer.transform()` calls specifically so it can be
swapped in a few lines:

```python
# embed_store.py — replace TfidfVectorizer with:
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
matrix = model.encode(texts, normalize_embeddings=True)

# retrieve.py — replace vectorizer.transform([query]) with:
query_vec = model.encode([query], normalize_embeddings=True)
```
(cosine similarity works the same way for normalized dense vectors.)

### Storage/index for embeddings
`joblib`-serialized files on local disk (`index/vectorizer.joblib`,
`index/matrix.joblib`). At 11 chunks, a dedicated vector database (FAISS,
Chroma, Pinecone) would be pure overhead — brute-force cosine similarity
over an 11-row matrix is sub-millisecond. This would need to change if the
corpus grew past a few thousand chunks or needed to be served
concurrently by multiple processes.

### LLM and prompt design
Claude via the Anthropic API. The system prompt does three things
explicitly: (1) restricts the model to the supplied context, (2) instructs
it to say so if the context doesn't contain the answer rather than
guessing, (3) caps answer length to keep responses "short and clear" as
required. Each retrieved chunk is labelled with its section title in the
prompt (`[Source: ...]`) so the model's answer can be checked against a
named source, and so the CLI can independently display which sources were
used (retrieved-and-shown, not model-self-reported).

## 2. What had to be learned/researched

- Confirming PIB backgrounder pages render as fairly clean semantic HTML
  with markdown-convertible headers, which made section-aligned chunking
  straightforward rather than requiring heavier HTML-structure parsing.
- Checking the sandbox's actual network allowlist before assuming a
  Hugging Face model download would work — this shaped the embedding
  choice above rather than being discovered mid-implementation.

## 3. Limitations and what I'd improve with 2 more days

1. **Swap in a real neural embedding model** (sentence-transformers or an
   API-based one) and run a small labelled eval set (10-15 question/answer
   pairs with the expected source chunk) to quantify retrieval accuracy
   TF-IDF vs. embeddings, rather than relying on spot checks.
2. **Add re-ranking**: retrieve top-10 by TF-IDF, then re-rank with a
   cross-encoder or the LLM itself for better precision at k=3.
3. **Handle multi-hop questions** (e.g. "compare AB-PMJAY and PM-ABHIM
   funding") that need two chunks combined — current top-k retrieval
   handles this only incidentally.
4. **Chunk-boundary evaluation**: automatically flag chunks whose word
   count falls outside 200-500 (one chunk currently sits at 168) and
   decide case-by-case whether to merge or accept, instead of a single
   fixed threshold.
5. **Add a citation-checking pass**: verify the LLM's answer's factual
   claims (numbers, dates) appear verbatim in the cited chunk text, to
   catch subtle hallucination the prompt-level constraint might miss.
6. **Swap CLI for a minimal web UI** (Streamlit/Flask) if a visual
   interface is preferred over CLI for demo purposes.
