"""
Extracts text and tables from any PDF (URL or local file path).
Outputs a list of page chunks with metadata.
"""

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pdfplumber
import requests
from rich.console import Console
from rich.progress import track

console = Console()

CHUNKS_PATH = Path("data/chunks.json")


def resolve_pdf(source: str) -> Path:
    """
    Accept a URL or a local file path and return a local Path to the PDF.
    Downloads the file if it's a URL.
    """
    if source.startswith("http://") or source.startswith("https://"):
        # Derive a safe filename from the URL (strip query params and fragments)
        parsed = urlparse(source)
        filename = parsed.path.rstrip("/").split("/")[-1] or "document"
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        dest = Path("data") / filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            console.print(f"[green]PDF already exists at {dest}[/green]")
            return dest

        console.print(f"[cyan]Downloading PDF from {source}...[/cyan]")
        response = requests.get(source, stream=True, timeout=60)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        console.print(f"[green]Downloaded to {dest}[/green]")
        return dest
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {source}")
        return path


def table_to_markdown(table: list[list]) -> str:
    if not table or not table[0]:
        return ""

    rows = []
    header = [str(cell).strip() if cell else "" for cell in table[0]]
    rows.append("| " + " | ".join(header) + " |")
    rows.append("| " + " | ".join(["---"] * len(header)) + " |")

    for row in table[1:]:
        cells = [str(cell).strip().replace("\n", " ") if cell else "" for cell in row]
        while len(cells) < len(header):
            cells.append("")
        rows.append("| " + " | ".join(cells[: len(header)]) + " |")

    return "\n".join(rows)


def extract_chunks(pdf_path: Path) -> list[dict[str, Any]]:
    chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        console.print(f"[cyan]Processing {total_pages} pages...[/cyan]")

        for page_num, page in enumerate(
            track(pdf.pages, description="Extracting pages..."), start=1
        ):
            tables = page.extract_tables()
            table_count = 0

            for table in tables:
                md = table_to_markdown(table)
                if not md or len(md) < 20:
                    continue
                chunks.append(
                    {
                        "chunk_id": f"page{page_num}_table{table_count}",
                        "page": page_num,
                        "type": "table",
                        "content": md,
                    }
                )
                table_count += 1

            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            text = text.strip()

            if text:
                text = re.sub(r"-\n(\w)", r"\1", text)
                text = re.sub(r"\n{3,}", "\n\n", text)
                chunks.append(
                    {
                        "chunk_id": f"page{page_num}_text",
                        "page": page_num,
                        "type": "text",
                        "content": text,
                    }
                )

    console.print(f"[green]Extracted {len(chunks)} raw chunks[/green]")
    return chunks


def semantic_split(chunks: list[dict], max_tokens: int = 500, overlap: int = 50) -> list[dict]:
    result = []
    words_per_token = 0.75

    for chunk in chunks:
        if chunk["type"] == "table":
            result.append(chunk)
            continue

        words = chunk["content"].split()
        max_words = int(max_tokens * words_per_token)
        overlap_words = int(overlap * words_per_token)

        if len(words) <= max_words:
            result.append(chunk)
            continue

        start = 0
        sub_idx = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            sub_text = " ".join(words[start:end])
            result.append(
                {
                    "chunk_id": f"{chunk['chunk_id']}_sub{sub_idx}",
                    "page": chunk["page"],
                    "type": "text",
                    "content": sub_text,
                }
            )
            sub_idx += 1
            start += max_words - overlap_words

    console.print(f"[green]After splitting: {len(result)} chunks[/green]")
    return result


def run_extraction(source: str) -> list[dict]:
    pdf_path = resolve_pdf(source)
    raw_chunks = extract_chunks(pdf_path)
    chunks = semantic_split(raw_chunks)

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    console.print(f"[green]Saved chunks to {CHUNKS_PATH}[/green]")
    return chunks
