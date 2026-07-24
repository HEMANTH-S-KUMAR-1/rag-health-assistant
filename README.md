# RAG Q&A Assistant — India's Health Transformation

A small end-to-end Retrieval-Augmented Generation (RAG) system that answers
questions about India's health transformation, grounded in a single PIB
backgrounder page:
https://www.pib.gov.in/PressReleasePage.aspx?PRID=2269699&reg=48&lang=2

## Pipeline overview

```
data/raw_page.md  --ingest.py-->  data/chunks.json  --embed_store.py-->  index/embeddings.npz
                                                                              |
                                                                              v
                                            user question --retrieve.py--> top-k chunks
                                                                              |
                                                                              v
                                                     generate.py --> grounded answer
                                                              |
                                                              v
                                                     verify.py --> citation check
```

## 1. Ingestion & chunking (`src/ingest.py`)

- The PIB page was fetched and saved as clean markdown at `data/raw_page.md`
  (images, boilerplate, and the reference-link list were stripped; only the
  substantive article text was kept).
- Chunking is **section-aligned, not fixed-size**: the script splits on the
  page's own `##`/`###` headers (AB-PMJAY, Ayushman Arogya Mandirs,
  PM-ABHIM, ABDM, individual NHM programmes, etc.), so each chunk stays
  topically coherent.
- Sections under 200 words are merged forward into the next section (their
  titles are concatenated, e.g. `"Pillar 3 ... / Pillar 4 ..."`, so the
  merge is visible rather than silently dropped).
- Sections over 500 words are split at paragraph boundaries, keeping each
  resulting piece as close to the 200–500 word band as possible without
  breaking a paragraph mid-sentence.
- Undersized trailing chunks are rebalanced with the preceding chunk using
  an even-split algorithm, keeping both chunks within the 200–500 band.
- Result: 19 chunks, word counts ranging 209–391 (all within the 200–500
  target band).

Run: `python src/ingest.py`

## 2. Semantic search (`src/embed_store.py`, `src/retrieve.py`)

- **Embedding model: `all-mpnet-base-v2`** (sentence-transformers, 768-dim).
  This is a high-quality general-purpose sentence embedding model that
  handles paraphrased queries well (unlike TF-IDF, which is lexical only).
- **Storage**: normalised embedding vectors stored as compressed NumPy arrays
  (`index/embeddings.npz`) — no external vector database is needed at this
  scale (19 chunks). Brute-force cosine similarity over a 19×768 matrix
  is sub-millisecond.
- **Search**: the question is encoded with the same model, then ranked
  against all chunk vectors by cosine similarity
  (`sklearn.metrics.pairwise.cosine_similarity`); the top-k are returned.
- **Multi-hop detection**: comparison-style questions (e.g. "compare
  AB-PMJAY and PM-ABHIM") are detected and split into sub-queries, with
  top-1 retrieved for each entity, then combined.
- **Optional LLM re-ranking**: with `--rerank`, the CLI retrieves a broader
  pool of candidates and asks Gemini to re-score them for relevance,
  improving precision at k=3.

Run: `python src/embed_store.py`

## 3. RAG answer generation (`src/generate.py`)

- LLM: Google Gemini API (`gemini-flash-lite-latest`, via `google-genai` SDK).
- The prompt includes: a system instruction constraining the model to
  answer only from the supplied context and say so explicitly if the
  context doesn't contain the answer, plus the user's question and the
  retrieved chunks (each labelled with its source title).
- This directly targets the "don't hallucinate beyond the document"
  requirement — grounding is enforced at the prompt level, not just by
  what's retrieved.

## 4. Citation verification (`src/verify.py`)

- After generating an answer, the system extracts verifiable claims
  (numbers, dates, percentages, currency amounts, scheme names) and
  checks each against the retrieved chunk text.
- Claims found verbatim in the source are marked as verified; others
  are flagged as unverified, catching subtle hallucinations the prompt
  constraint alone might miss.

## 5. Evaluation (`src/evaluate.py`)

- A labeled eval set of 15 questions (`data/eval_set.json`) covering
  exact-vocabulary queries, paraphrased queries, and negative queries.
- Reports Hit@1, Hit@3, and Mean Reciprocal Rank (MRR) metrics.

Run: `python src/evaluate.py`

## 6. Interface (`src/cli.py`, `src/app.py`)

**CLI** — Type a question, see the answer, see which chunks were used
with similarity scores, and see citation verification results.

**Web UI** — A Flask-based web interface with a dark-themed, modern
design. Ask questions in a browser, see answers with source cards
and citation verification badges.

## Setup

```bash
git clone <this-repo-url>
cd rag-health-assistant

# create and activate a virtual environment
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\activate

pip install -r requirements.txt

# create your .env from the provided example
cp .env.example .env
# edit .env and paste your Gemini API key
export $(cat .env | xargs)          # Linux / macOS
# Windows (PowerShell): $env:GEMINI_API_KEY="<your key>"

python src/ingest.py        # data/raw_page.md -> data/chunks.json
python src/embed_store.py   # data/chunks.json -> index/embeddings.npz
```

## Run

```bash
# CLI — interactive
python src/cli.py

# CLI — single question
python src/cli.py -q "How many Ayushman Arogya Mandirs are functional?"

# CLI — with LLM re-ranking
python src/cli.py -q "Compare AB-PMJAY and PM-ABHIM" --rerank

# Web UI
python src/app.py
# then open http://127.0.0.1:5000

# Evaluate retrieval quality
python src/evaluate.py
```

Example output:

```
======================================================================
Q: How many Ayushman Arogya Mandirs are functional?
----------------------------------------------------------------------
As of the document, over 1.86 lakh Ayushman Arogya Mandirs are functional,
including 1.34 lakh Sub Health Centres, 24,483 Primary Health Centres,
5,474 Urban Primary Health Centres, 12,259 AYUSH centres, and 9,758 Urban
Health and Wellness Centres.
----------------------------------------------------------------------
Sources used:
  [0.645] Pillar 2: Primary Care Through Ayushman Arogya Mandirs (AAM)
          "The government is scaling primary healthcare infrastructure..."
----------------------------------------------------------------------
Citation check: 6/6 claims verified in source
======================================================================
```

## Project structure

```
rag-health-assistant/
├── data/
│   ├── raw_page.md       # cleaned source text (committed)
│   ├── chunks.json       # generated by ingest.py (gitignored)
│   └── eval_set.json     # labeled eval set (15 questions)
├── index/
│   └── embeddings.npz    # generated by embed_store.py (gitignored)
├── src/
│   ├── ingest.py         # step 1: chunking
│   ├── embed_store.py    # step 2: embeddings (all-mpnet-base-v2)
│   ├── retrieve.py       # step 3: search + re-ranking + multi-hop
│   ├── generate.py       # step 4: LLM answer generation
│   ├── verify.py         # citation verification
│   ├── evaluate.py       # retrieval eval metrics
│   ├── cli.py            # CLI interface
│   ├── app.py            # Flask web UI
│   └── templates/
│       └── index.html    # web UI template
├── scripts/
│   ├── extract_pdf.py    # one-off: PDF text extraction (local utility)
│   └── extract_images.py # one-off: OCR via Gemini vision (local utility)
├── requirements.txt
├── .env.example
├── IMPLEMENTATION_NOTE.md
└── README.md
```

> **Note:** `scripts/extract_pdf.py` and `scripts/extract_images.py` are one-off
> local utilities used to extract and OCR the source PDF into `data/raw_page.md`.
> They are not part of the runtime pipeline and contain hardcoded local paths.
