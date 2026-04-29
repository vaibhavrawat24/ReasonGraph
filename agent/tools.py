"""
Tool implementations for the agent.
Each tool has a JSON schema definition (for the LLM) and an execute function.
"""

import ast
import json
import math
import operator
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_PATH = Path("data/chroma_db")
COLLECTION_NAME = "cyber_ireland_2022"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

_collection: chromadb.Collection | None = None
_embed_model: SentenceTransformer | None = None


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


# ── Tool schemas (OpenAI function-call format) ─────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": (
                "Semantically search the Cyber Ireland 2022 report. "
                "Returns the top-k most relevant chunks with their page numbers. "
                "Use this as your first step for any query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query in natural language.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default 6, max 15).",
                        "default": 6,
                    },
                    "chunk_type": {
                        "type": "string",
                        "enum": ["text", "table", "any"],
                        "description": "Filter by chunk type. Use 'table' when looking for statistics or regional data.",
                        "default": "any",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page",
            "description": (
                "Retrieve all chunks (text and tables) from a specific page number. "
                "Use this to verify a citation or when you know which page to inspect."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "The page number to retrieve (1-indexed).",
                    }
                },
                "required": ["page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Safely evaluate a mathematical expression and return the result. "
                "Always use this for arithmetic, percentages, CAGR, or any numeric computation. "
                "Never attempt math in your head — use this tool instead. "
                "Supports: +, -, *, /, **, sqrt(), log(), round(), abs(), pow(). "
                "Example: '(19000 / 6500) ** (1/8) - 1' for CAGR."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid Python math expression as a string.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cite",
            "description": (
                "Record a verified citation from the document. "
                "Call this when you have found the exact location of a fact. "
                "Returns a structured citation object that will appear in the final answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "description": "The page number where the fact appears.",
                    },
                    "quote": {
                        "type": "string",
                        "description": "The exact verbatim text from the document that supports the fact.",
                    },
                    "fact": {
                        "type": "string",
                        "description": "A one-sentence summary of what this citation proves.",
                    },
                },
                "required": ["page", "quote", "fact"],
            },
        },
    },
]


# ── Tool implementations ───────────────────────────────────────────────────────

def tool_vector_search(query: str, k: int = 6, chunk_type: str = "any") -> dict:
    model = _get_embed_model()
    # BGE asymmetric retrieval prefix
    query_embedding = model.encode(
        f"Represent this sentence for searching relevant passages: {query}",
        normalize_embeddings=True,
    ).tolist()

    collection = _get_collection()

    where = None
    if chunk_type in ("text", "table"):
        where = {"type": chunk_type}

    k = min(k, 15)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "page": meta["page"],
                "type": meta["type"],
                "relevance_score": round(1 - dist, 4),
                "content": doc,
            }
        )

    return {"query": query, "results": hits}


def tool_get_page(page: int) -> dict:
    collection = _get_collection()
    results = collection.get(
        where={"page": page},
        include=["documents", "metadatas"],
    )

    chunks = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        chunks.append({"type": meta["type"], "content": doc})

    if not chunks:
        return {"page": page, "error": "No content found for this page."}

    return {"page": page, "chunks": chunks}


# Safe math evaluator — no eval() on arbitrary code
_SAFE_NAMES = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "pow": pow,
    "pi": math.pi,
    "e": math.e,
}

_ALLOWED_OPS = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd, ast.Name, ast.Load,
)


def _safe_eval(expr: str) -> float:
    tree = ast.parse(expr.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_OPS):
            raise ValueError(f"Disallowed expression node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in _SAFE_NAMES:
            raise ValueError(f"Unknown name: {node.id}")
    return eval(compile(tree, "<string>", "eval"), {"__builtins__": {}}, _SAFE_NAMES)


def tool_calculate(expression: str) -> dict:
    try:
        result = _safe_eval(expression)
        return {
            "expression": expression,
            "result": result,
            "result_formatted": f"{result:.6f}",
        }
    except Exception as e:
        return {"expression": expression, "error": str(e)}


def tool_cite(page: int, quote: str, fact: str) -> dict:
    return {
        "citation": {
            "page": page,
            "quote": quote,
            "fact": fact,
        }
    }


# ── Dispatcher ─────────────────────────────────────────────────────────────────

TOOL_MAP = {
    "vector_search": tool_vector_search,
    "get_page": tool_get_page,
    "calculate": tool_calculate,
    "cite": tool_cite,
}


def execute_tool(name: str, args: dict) -> Any:
    if name not in TOOL_MAP:
        return {"error": f"Unknown tool: {name}"}
    return TOOL_MAP[name](**args)
