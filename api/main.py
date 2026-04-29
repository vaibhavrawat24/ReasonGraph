"""
FastAPI backend exposing:
  POST /ingest        — ingest any PDF (URL or file path) into the vector store
  POST /query         — blocking, returns full result when done
  POST /query/stream  — SSE stream, emits agent steps in real time
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

from agent.orchestrator import run_agent, stream_agent
from etl.pdf_extractor import run_extraction
from etl.embedder import embed_and_store

DEFAULT_SOURCE = "https://cyberireland.ie/wp-content/uploads/2022/05/State-of-the-Cyber-Security-Sector-in-Ireland-2022-Report.pdf"

app = FastAPI(
    title="Document Intelligence API",
    description="Agentic RAG backend — ingest any PDF and query it with a reasoning agent.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class IngestRequest(BaseModel):
    source: str = DEFAULT_SOURCE  # URL or local file path


class QueryRequest(BaseModel):
    query: str


class CitationOut(BaseModel):
    page: int
    quote: str
    fact: str


class CalculationOut(BaseModel):
    expression: str
    result: float
    result_formatted: str


class QueryResponse(BaseModel):
    run_id: str
    query: str
    answer: str
    citations: list[CitationOut]
    calculations: list[CalculationOut]
    iterations: int
    elapsed_seconds: float
    model: str
    timestamp: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", status_code=202)
def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest a PDF into the vector store.
    Accepts a public URL or an absolute local file path.
    Runs in the background — poll /health or check server logs for completion.

    Examples:
      {"source": "https://example.com/report.pdf"}
      {"source": "/Users/me/documents/report.pdf"}
    """
    def _run():
        chunks = run_extraction(request.source)
        embed_and_store(chunks)

    background_tasks.add_task(_run)
    return {"status": "ingestion started", "source": request.source}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Blocking endpoint — waits for full agent completion before responding."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return run_agent(request.query)


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    """
    SSE streaming endpoint — emits agent steps in real time.

    Event types:
      start       — run started {run_id, query, model}
      iteration   — LLM call starting {iteration, status}
      tool_call   — tool being invoked {iteration, tool, args}
      tool_result — tool result preview {iteration, tool, result_preview}
      citation    — a citation was recorded {page, quote, fact}
      calculation — a calculation was performed {expression, result, result_formatted}
      done        — final answer {run_id, answer, citations, calculations, ...}
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return StreamingResponse(
        stream_agent(request.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
