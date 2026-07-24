# RAG Q&A Assistant — India's Health Transformation

A small end-to-end Retrieval-Augmented Generation (RAG) system that answers
questions about India's health transformation, grounded in a single PIB
backgrounder page:
https://www.pib.gov.in/PressReleasePage.aspx?PRID=2269699&reg=48&lang=2

## Pipeline overview

```
data/raw_page.md  --ingest.py-->  data/chunks.json  --embed_store.py-->  index/*.joblib
                                                                              |
                                                                              v
                                            user question --retrieve.py--> top-k chunks
                                                                              |
                                                                              v
                                                              generate.py --> grounded answer
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
- Result: 11 chunks, word counts ranging 168–434 (one chunk sits slightly
  under 200 words rather than being forced back over 500 — documented
  trade-off, see Limitations below).

Run: `python src/ingest.py`

## 2. Semantic search (`src/embed_store.py`, `src/retrieve.py`)

- **Embedding model: TF-IDF** (scikit-learn `TfidfVectorizer`, word
  1–2 grams, English stop words removed, sublinear TF scaling).
- **Why TF-IDF and not a neural embedding model**: this environment has no
  network path to Hugging Face or an embeddings API, so a neural model
  can't be downloaded/called here. TF-IDF is a legitimate, fully offline,
  deterministic embedding representation, and it performs well on this
  particular document because the content is dense with distinctive named
  entities (scheme names, acronyms) that TF-IDF weights strongly. See
  `IMPLEMENTATION_NOTE.md` for the one-file swap to
  `sentence-transformers` if you have unrestricted internet access.
- **Storage**: the fitted vectorizer and the resulting sparse TF-IDF matrix
  are persisted to disk with `joblib` (`index/vectorizer.joblib`,
  `index/matrix.joblib`) — no external vector database is needed at this
  scale (11 chunks).
- **Search**: the question is transformed with the same vectorizer, then
  ranked against all chunk vectors by cosine similarity
  (`sklearn.metrics.pairwise.cosine_similarity`); the top-k are returned.

Run: `python src/embed_store.py`

## 3. RAG answer generation (`src/generate.py`)

- LLM: Claude (Anthropic API, `anthropic` Python SDK).
- The prompt includes: a system instruction constraining the model to
  answer only from the supplied context and say so explicitly if the
  context doesn't contain the answer, plus the user's question and the
  retrieved chunks (each labelled with its source title).
- This directly targets the "don't hallucinate beyond the document"
  requirement — grounding is enforced at the prompt level, not just by
  what's retrieved.

## 4. Interface (`src/cli.py`)

A CLI that:
- Takes a question (interactively or via `--question`)
- Prints the generated answer
- Prints which chunks were used, with similarity scores and a short
  snippet of each, so answers are auditable against the source

## Setup

```bash
git clone <this-repo-url>
cd rag-health-assistant
pip install -r requirements.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=<your key>
export $(cat .env | xargs)

python src/ingest.py        # data/raw_page.md -> data/chunks.json
python src/embed_store.py   # data/chunks.json -> index/*.joblib
```

## Run

```bash
# interactive
python src/cli.py

# single question
python src/cli.py -q "How many Ayushman Arogya Mandirs are functional?"

# retrieve more chunks
python src/cli.py -q "What has the government done for tuberculosis?" --top-k 5
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
  [0.301] Pillar 2: Primary Care Through Ayushman Arogya Mandirs (AAM)
          "The government is scaling primary healthcare infrastructure..."
======================================================================
```

## Project structure

```
rag-health-assistant/
├── data/
│   ├── raw_page.md       # cleaned source text (committed)
│   └── chunks.json       # generated by ingest.py (gitignored)
├── index/
│   ├── vectorizer.joblib # generated by embed_store.py (gitignored)
│   └── matrix.joblib
├── src/
│   ├── ingest.py
│   ├── embed_store.py
│   ├── retrieve.py
│   ├── generate.py
│   └── cli.py
├── requirements.txt
├── .env.example
└── README.md
```
