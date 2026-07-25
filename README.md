# RAG Q&A Assistant — India's Health Transformation

A Retrieval-Augmented Generation pipeline that answers questions about
India's health-sector programmes, grounded entirely in a single PIB
backgrounder document. Built with sentence-transformers for semantic
search and Google Gemini for answer generation.

---

## Setup

```bash
git clone https://github.com/HEMANTH-S-KUMAR-1/rag-health-assistant.git
cd rag-health-assistant

# Create and activate a virtual environment
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\activate

pip install -r requirements.txt

# Create your .env from the provided example and add your Gemini API key
cp .env.example .env          # Linux / macOS
copy .env.example .env        # Windows
# Edit .env and paste your GEMINI_API_KEY
```

> **Note:** `data/chunks.json` and `index/embeddings.npz` are committed
> to the repo, so you can skip straight to running the CLI or web UI.
> Re-run `python src/ingest.py` and `python src/embed_store.py` only if
> you want to re-chunk or re-embed from scratch.

---

## Run

```bash
# Interactive CLI
python src/cli.py

# Single question
python src/cli.py -q "What is AB-PMJAY?"

# With LLM-based re-ranking
python src/cli.py -q "Compare AB-PMJAY and PM-ABHIM" --rerank

# Web UI (opens at http://127.0.0.1:5000)
python src/app.py

# Evaluate retrieval quality
python src/evaluate.py
```

---

## Pipeline overview

```
data/raw_page.md ──ingest.py──► data/chunks.json ──embed_store.py──► index/embeddings.npz
                                                                           │
                                                                           ▼
                                        user question ──retrieve.py──► top-k chunks
                                                                           │
                                                                           ▼
                                                    generate.py ──► grounded answer
                                                                           │
                                                                           ▼
                                                      verify.py ──► citation check
```

### 1. Ingestion & chunking (`src/ingest.py`)

The source document (`data/raw_page.md`) is a PIB backgrounder on
India's health transformation, already converted to Markdown.

Chunking is **section-aligned, not fixed-size**: the script splits on the
document's own `##`/`###` headers (AB-PMJAY, Ayushman Arogya Mandirs,
PM-ABHIM, ABDM, individual NHM sub-programmes, etc.), so each chunk
stays topically coherent.

A three-step merge–split–redistribute pass enforces chunk-size constraints:

| Step | What it does |
|---|---|
| **Merge** | Adjacent small sections (< 200 words) are merged under a combined title (e.g. `"Pillar 3 … / Pillar 4 …"`). |
| **Split** | Any merged section exceeding 500 words is split at paragraph boundaries into ≤ 500-word pieces. |
| **Redistribute** | If splitting leaves a trailing runt chunk (< 200 words), paragraphs from the last two chunks are redistributed at the closest paragraph boundary so both meet the minimum. |

**Result:** 19 chunks, 209–391 words each, all within the 200–500-word
target band. Saved as `data/chunks.json` (array of `{id, title, text, word_count}`).

### 2. Embedding & storage (`src/embed_store.py`)

Each chunk's text is encoded with
[`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
(768-dimensional, normalised embeddings). The resulting 19 × 768 matrix
is saved as `index/embeddings.npz` — a compressed NumPy archive.
Brute-force cosine similarity over this matrix is sub-millisecond; no
vector-DB overhead is needed at this scale.

### 3. Retrieval (`src/retrieve.py`)

1. The user query is embedded with the same model.
2. Cosine similarities are computed against all chunk embeddings
   (`sklearn.metrics.pairwise.cosine_similarity`).
3. Top-k chunks are returned (default k = 3).

**Multi-hop detection:** comparison-style queries (containing "compare",
"vs", "between", or multiple "and" conjunctions) are split into
sub-queries, each retrieving top results independently, so both relevant
chunks surface.

**Optional LLM re-ranking** (`--rerank` flag): the top-8 embedding
candidates are sent to Gemini with a relevance-rating prompt; chunks are
re-sorted by the LLM's 0–10 score before the final top-k are selected.

### 4. Answer generation (`src/generate.py`)

Top-k chunks are injected into a system prompt that instructs Gemini to:
- answer **only** from the provided context,
- say "I don't have enough information" if the context is insufficient,
- keep answers short and direct (2–5 sentences).

The model used is **`gemini-flash-lite-latest`**, called via the
`google-genai` Python SDK.

### 5. Citation verification (`src/verify.py`)

After generation, verifiable claims (numbers, dates, percentages, ₹
amounts) are extracted from the answer and checked against the source
chunk text. Results are shown in both the CLI and web UI.

### 6. Evaluation (`src/evaluate.py`)

`data/eval_set.json` contains 15 question–answer pairs (7 exact, 7
paraphrased, 1 negative) with expected chunk titles. The evaluator runs
retrieval for each question and reports Hit@1, Hit@3, and MRR.

---

## Project structure

```
rag-health-assistant/
├── data/
│   ├── raw_page.md        # source document (committed)
│   ├── chunks.json        # 19 chunked sections (committed)
│   └── eval_set.json      # 15 evaluation questions
├── index/
│   └── embeddings.npz     # 19 × 768 embedding matrix (committed)
├── src/
│   ├── ingest.py          # step 1: chunk raw_page.md
│   ├── embed_store.py     # step 2: embed chunks → .npz
│   ├── retrieve.py        # step 3: semantic search + re-rank + multi-hop
│   ├── generate.py        # step 4: Gemini RAG generation
│   ├── verify.py          # citation verification
│   ├── evaluate.py        # retrieval evaluation (Hit@k, MRR)
│   ├── cli.py             # interactive terminal Q&A
│   ├── app.py             # Flask web UI
│   └── templates/
│       └── index.html     # web UI template
├── scripts/               # one-off local data-preparation utilities (not runtime)
│   ├── extract_pdf.py
│   ├── extract_images.py
│   └── README.md
├── .env.example
├── requirements.txt
├── IMPLEMENTATION_NOTE.md
└── README.md
```

---

## Requirements

- Python 3.9+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey) (free tier works)
- See `requirements.txt` for Python packages
