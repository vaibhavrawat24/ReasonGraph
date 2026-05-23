"""
Embeds chunks using sentence-transformers and stores them in ChromaDB.
"""

import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import track
import agent.tools as _agent_tools

console = Console()

CHUNKS_PATH = Path("data/chunks.json")
CHROMA_PATH = Path("data/chroma_db")
COLLECTION_NAME = "reasongraph_docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def embed_and_store(chunks: list[dict[str, Any]] | None = None) -> chromadb.Collection:
    if chunks is None:
        with open(CHUNKS_PATH) as f:
            chunks = json.load(f)

    # Delete and recreate so a fresh ingest always replaces the old document.
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(COLLECTION_NAME)
        console.print(f"[yellow]Dropped existing collection '{COLLECTION_NAME}'[/yellow]")
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    console.print(f"[cyan]Loading embedding model: {EMBED_MODEL}[/cyan]")
    model = SentenceTransformer(EMBED_MODEL)

    # BGE models need a query prefix for asymmetric retrieval
    texts = [c["content"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"page": c["page"], "type": c["type"]} for c in chunks]

    console.print(f"[cyan]Embedding {len(chunks)} chunks...[/cyan]")

    batch_size = 64
    all_embeddings = []
    for i in track(range(0, len(texts), batch_size), description="Embedding batches..."):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, normalize_embeddings=True).tolist()
        all_embeddings.extend(embeddings)

    console.print("[cyan]Upserting into ChromaDB...[/cyan]")

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            embeddings=all_embeddings[i : i + batch_size],
            documents=texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    console.print(f"[green]Stored {collection.count()} chunks in ChromaDB[/green]")

    # Reset cached singleton so the next query fetches the fresh collection.
    _agent_tools._collection = None

    return collection


def run_embed():
    embed_and_store()


if __name__ == "__main__":
    run_embed()
