"""
Main ETL entry point. Run this once before starting the API.

Usage:
  python -m etl.run_etl                          # uses built-in default PDF
  python -m etl.run_etl path/to/file.pdf         # local file
  python -m etl.run_etl https://example.com/doc.pdf  # remote URL
"""

import sys
from rich.console import Console
from rich.rule import Rule
from etl.pdf_extractor import run_extraction
from etl.embedder import embed_and_store

console = Console()

DEFAULT_SOURCE = "https://cyberireland.ie/wp-content/uploads/2022/05/State-of-the-Cyber-Security-Sector-in-Ireland-2022-Report.pdf"


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE

    console.print(Rule("[bold cyan]PDF Ingestion Pipeline[/bold cyan]"))
    console.print(f"[dim]Source: {source}[/dim]\n")

    console.print("[bold]Step 1: Extract PDF[/bold]")
    chunks = run_extraction(source)

    console.print("\n[bold]Step 2: Embed & Store[/bold]")
    collection = embed_and_store(chunks)

    console.print(f"\n[bold green]ETL complete. {collection.count()} chunks ready.[/bold green]")


if __name__ == "__main__":
    main()
