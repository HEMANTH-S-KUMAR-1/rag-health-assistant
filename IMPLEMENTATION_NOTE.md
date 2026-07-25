# Implementation Note — RAG Q&A Assistant (India's Health Transformation)

## 1. Choices

### Chunking strategy

Header-aligned splitting rather than a fixed-size sliding window.  The
source document is already organised under named sections (AB-PMJAY,
Ayushman Arogya Mandirs, PM-ABHIM, ABDM, NHM sub-programmes), and the
assignment explicitly asks for chunks that map to those sections.  A
sliding window would frequently cut a scheme's description in half.  The
trade-off: section lengths in the source are uneven, so a rule-based
merge/split step was needed to keep every chunk within 200–500 words
while preserving topic boundaries.

A balanced-redistribution algorithm handles undersized trailing chunks:
when splitting a long section produces a runt chunk under 200 words, the
paragraphs in the last two chunks are redistributed more evenly at the
closest paragraph boundary.  Result: **19 chunks, 209–391 words**, all
within the 200–500 target band.

### Embedding model: `all-mpnet-base-v2` (sentence-transformers)

`all-mpnet-base-v2` produces 768-dimensional normalised embeddings with
strong semantic quality (~63 on the STS benchmark).  It handles
paraphrased and synonym-heavy queries that lexical methods like TF-IDF
would miss entirely (e.g. "elderly healthcare" correctly maps to the
AB-PMJAY chunk about "senior citizens above 70 years").

Why `all-mpnet-base-v2` specifically and not `all-MiniLM-L6-v2`: mpnet
is a higher-quality model with more dimensions (768 vs 384) and better
benchmark scores.  At 19 chunks, the extra compute is negligible, so the
quality trade-off favours the larger model.

### Storage / index for embeddings

NumPy `.npz` compressed arrays on local disk (`index/embeddings.npz`).
At 19 chunks × 768 dimensions, a dedicated vector database (FAISS,
Chroma, Pinecone) would be pure overhead — brute-force cosine similarity
over this matrix is sub-millisecond.  The storage format is standard,
portable, and readable by any NumPy installation.  This would need to
change if the corpus grew past a few thousand chunks or needed concurrent
serving.

### LLM and prompt design

Google Gemini (`gemini-flash-lite-latest`) via the `google-genai` Python
SDK.  The system prompt does three things explicitly:

1. Restricts the model to the supplied context.
2. Instructs it to say so if the context doesn't contain the answer
   rather than guessing.
3. Caps answer length to keep responses "short and direct" (2–5
   sentences) as required.

Each retrieved chunk is labelled with its source title so the model can
cite sections.

### Re-ranking

An optional two-stage retrieval pipeline: first, retrieve the top-8
candidates by embedding cosine similarity; then, ask Gemini to rate each
passage's relevance on a 0–10 scale and re-sort by that score.  This
cross-encoder-style re-ranking improves precision at k = 3 without
touching the embedding layer.  Enabled via a `--rerank` CLI flag; falls
back to pure embedding ranking when no API key is set.

### Multi-hop question handling

Comparison-style questions ("compare AB-PMJAY and PM-ABHIM funding") are
detected by pattern matching (keywords: "compare", "vs", "between",
multiple conjunctions).  For these, the query is split into sub-queries,
top results are retrieved for each entity, and results are combined —
ensuring both relevant chunks surface rather than relying on accidental
co-ranking.

### Citation verification

A post-generation verification pass extracts verifiable claims (numbers,
dates, percentages, currency amounts) from the generated answer and
checks each against the source chunk text.  This catches subtle
hallucinations that the prompt-level "answer only from context"
constraint alone might miss.  Results are displayed in both the CLI and
web UI.

---

## 2. What had to be learned / researched

- Confirming PIB backgrounder pages render as fairly clean semantic HTML
  with markdown-convertible headers, which made section-aligned chunking
  straightforward rather than requiring heavier HTML-structure parsing.
- Evaluating sentence-transformers model options: compared
  `all-MiniLM-L6-v2` (faster, 384-dim) vs `all-mpnet-base-v2` (higher
  quality, 768-dim) on the STS benchmark and chose the latter given the
  small corpus size.
- Designing a balanced-redistribution algorithm for the chunk boundary
  edge case where a merged+split section produces a sub-200-word runt.
- Structuring an eval set that systematically tests both exact-vocabulary
  and paraphrased queries, with clear expected-chunk labels for automated
  scoring.

---

## 3. Limitations and what I'd improve with 2 more days

1. **Hybrid retrieval**: combine embedding cosine similarity with BM25
   keyword scores (reciprocal rank fusion) for the best of both worlds —
   semantic understanding plus exact keyword matching.
2. **Chunk overlap**: add 1–2 sentence overlap between adjacent chunks
   from the same section, so information that spans a paragraph boundary
   isn't lost.
3. **Streaming answers**: switch from synchronous API calls to Gemini's
   streaming API so the web UI can display answers token-by-token, giving
   a more responsive feel.
4. **Unit tests**: add pytest-based tests for each pipeline stage
   (chunking, embedding, retrieval, verification) to catch regressions.
5. **Confidence thresholds**: when all retrieved chunks have low
   similarity scores (< 0.2), proactively tell the user the question may
   not be covered by the source, rather than generating a potentially
   low-quality answer.
6. **Multi-document support**: extend the pipeline to handle multiple PIB
   backgrounders or policy documents, with document-level metadata in the
   chunk structure.
