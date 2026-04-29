# ReasonGraph — Agentic Document Intelligence Backend

A headless Python backend that ingests any PDF, processes it into a queryable vector store, and uses an autonomous LLM agent to answer complex multi-step queries with verifiable citations, grounded math, and exact page references.

---

## Architecture

```
PDF → pdfplumber (extract text + tables) → semantic chunker
    → sentence-transformers (BAAI/bge-small-en-v1.5) → ChromaDB

User query → FastAPI /query → Agent Orchestrator
    → Tool loop (vector_search / get_page / calculate / cite)
    → Gemini 2.5 Pro (via OpenRouter)
    → Structured response + JSON trace log
```

### Agent Tools

| Tool | Purpose |
|------|---------|
| `vector_search` | Semantic similarity search over ChromaDB |
| `get_page` | Fetch all chunks from a specific page number |
| `calculate` | Safe AST-based math evaluator (no LLM arithmetic) |
| `cite` | Force grounded citation with page + verbatim quote |

---

## Requirements

- **Python 3.11 or 3.12** (PyTorch does not support 3.13+ yet)
- An [OpenRouter](https://openrouter.ai) API key

---

## Setup

### 1. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First install pulls PyTorch + sentence-transformers (~1.5 GB). Takes 2–5 minutes depending on your connection.

### 3. Set your API key

```bash
cp .env.example .env
# Open .env and set your OpenRouter API key
```

`.env` contents:
```
OPENROUTER_API_KEY=sk-or-v1-...
```

### 4. Run the ETL pipeline (once)

The pipeline accepts any PDF — a remote URL or a local file path:

```bash
# Default: Cyber Ireland 2022 report (bundled example)
python -m etl.run_etl

# Any public URL
python -m etl.run_etl https://example.com/your-report.pdf

# Local file
python -m etl.run_etl /path/to/your-report.pdf
```

This will:
- Download the PDF if a URL is provided (~depends on file size)
- Extract text and tables from every page
- Download the embedding model on first run (~130 MB, cached after that)
- Embed all chunks and store them in `data/chroma_db/`

Expect ~2–3 minutes on first run (model download), ~30 seconds on subsequent runs.

> **Re-ingesting a new document** will overwrite the existing vector store. To support multiple documents simultaneously, run separate instances with different `data/` directories.

### 5. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

API is now live at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 6. Run the three evaluation tests

```bash
python run_tests.py
```

This fires all 3 graded queries in sequence and saves results + traces to `logs/`.

---

## API Usage

### POST `/query` — blocking

Waits for the full agent response before returning.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the total number of jobs reported?"}'
```

**Response:**

```json
{
  "run_id": "a1b2c3d4",
  "query": "What is the total number of jobs reported?",
  "answer": "The report states a total of 7,351 jobs...",
  "citations": [
    {
      "page": 19,
      "quote": "Total: | 489 | | 7,351",
      "fact": "Total employment in the cybersecurity sector is 7,351"
    }
  ],
  "calculations": [],
  "iterations": 3,
  "elapsed_seconds": 22.4,
  "model": "google/gemini-2.5-pro-preview",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### POST `/query/stream` — Server-Sent Events

Streams agent steps in real time so you can see progress as it works.

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the total number of jobs reported?"}'
```

**Event stream:**

```
event: start
data: {"run_id": "a1b2c3d4", "query": "...", "model": "google/gemini-2.5-pro-preview"}

event: iteration
data: {"iteration": 1, "status": "Calling LLM..."}

event: tool_call
data: {"iteration": 1, "tool": "vector_search", "args": {"query": "total number of jobs"}}

event: tool_result
data: {"iteration": 1, "tool": "vector_search", "result_preview": "..."}

event: citation
data: {"page": 19, "quote": "...", "fact": "..."}

event: done
data: {"run_id": "...", "answer": "...", "citations": [...], "calculations": [...], ...}
```

| Event | When | Contents |
|-------|------|----------|
| `start` | Immediately | run_id, query, model |
| `iteration` | Each LLM call | iteration number, status |
| `tool_call` | Before each tool | tool name + args |
| `tool_result` | After each tool | preview of result |
| `citation` | When agent cites a fact | page, quote, fact |
| `calculation` | When agent does math | expression, result |
| `done` | Final | full answer + all citations |

### POST `/ingest` — load a new document

Triggers ingestion of any PDF in the background. No need to restart the server.

```bash
# Remote URL
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "https://example.com/report.pdf"}'

# Local file
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "/path/to/report.pdf"}'
```

Returns `202 Accepted` immediately. Ingestion runs in the background — check server logs for completion.

### GET `/health`

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Architecture Justification

### ETL Strategy

**`pdfplumber` for extraction** — handles both narrative text and tables natively. Tables are extracted as structured lists and converted to Markdown, preserving relational data. Narrative text is split into ~500-token chunks with 50-token overlap to balance context window size and retrieval precision.

**`BAAI/bge-small-en-v1.5` for embeddings** — top-performing small model on the BEIR benchmark. Runs entirely locally (no API cost), fast on CPU, and uses asymmetric query prefixing for better retrieval accuracy on factual queries.

**ChromaDB** — zero-infrastructure persistent vector store. Ideal for single-document use cases; trivially swappable for Pinecone or Weaviate at scale.

### Agent Framework

**Raw OpenAI-format tool-use loop** (not LangChain/LlamaIndex) — chosen for full observability. Every tool call, argument, and result is captured in the JSON trace. LangChain abstractions hide reasoning steps, which conflicts with the grading criteria requiring proof of the agent's thought process.

**Gemini 2.5 Pro via OpenRouter** — strongest reasoning and tool-use model available on OpenRouter. Excels at multi-step planning and table comprehension, which is critical for the regional data synthesis query.

**Mandatory `calculate` tool** — LLMs are unreliable at arithmetic. All numeric operations (CAGR, percentages, comparisons) are routed through a safe AST-based evaluator, guaranteeing mathematically correct results for Test 3.

**Mandatory `cite` tool** — forces the agent to ground every factual claim in a page number and verbatim quote before concluding. The agent cannot return a final answer without having called `cite()`, eliminating hallucinations for Test 1.

---

## Limitations & Production Scaling

### Current Weaknesses

| Issue | Impact |
|-------|--------|
| `pdfplumber` struggles with complex merged/nested table cells | Some regional table values may be misaligned |
| BGE small model has lower recall than larger embedding models | May miss relevant chunks on ambiguous queries |
| No query re-ranking (cross-encoder) | Top-k results may not always be the most relevant |
| Single PDF, no incremental ingestion | Requires full re-ETL for document updates |
| No authentication on the API | Not production-safe as-is |
| Gemini 2.5 Pro is slow (~20–60s per query) | Poor UX without the streaming endpoint |

### Production Scaling Path

- **Embedding**: Swap to `text-embedding-3-large` (OpenAI) or `voyage-large-2` for higher recall
- **Vector DB**: Migrate ChromaDB → Pinecone/Weaviate with metadata filtering and namespacing per document
- **Re-ranking**: Add a cross-encoder (e.g., `ms-marco-MiniLM`) to re-rank top-20 → top-5 before LLM context
- **Caching**: Cache embeddings and frequent query results (Redis)
- **Observability**: Instrument with LangSmith or Langfuse for production trace monitoring and cost tracking
- **Auth**: Add API key middleware or OAuth2 to the FastAPI layer
- **Multi-document**: Extend ETL with a document registry and per-document metadata filtering in ChromaDB

---

## Project Structure

```
.
├── etl/
│   ├── pdf_extractor.py   # PDF download, text + table extraction, chunking
│   ├── embedder.py        # Embedding + ChromaDB ingestion
│   └── run_etl.py         # ETL entry point
├── agent/
│   ├── tools.py           # Tool schemas + implementations
│   └── orchestrator.py    # Agentic tool-use loop, streaming + blocking modes
├── api/
│   └── main.py            # FastAPI app (/query and /query/stream)
├── data/                  # PDF, chunks.json, chroma_db/ (git-ignored)
├── logs/                  # JSON trace logs per run (git-ignored)
├── run_tests.py           # Fires all 3 evaluation queries
├── requirements.txt
└── .env.example
```
