"""
Agentic orchestrator: manages the tool-use loop with Gemini 2.5 Pro via OpenRouter.
Supports both blocking (run_agent) and streaming (stream_agent) modes.
"""

import json
import os
import time
import uuid
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from agent.tools import TOOL_SCHEMAS, execute_tool

console = Console()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemini-2.5-pro-preview"
MAX_ITERATIONS = 12
LOGS_DIR = Path("logs")

SYSTEM_PROMPT = """You are a precise research analyst for the ingested document.
Your job is to answer queries with complete accuracy and full citations.

Rules you must follow:
1. Always start with vector_search to find relevant content.
2. Use get_page to verify exact wording before citing.
3. NEVER do arithmetic in your head — always call calculate() for any math.
4. Always call cite() for every factual claim with its exact page and verbatim quote.
5. If your first search doesn't find enough detail, try at least 2-3 different search queries before concluding.
6. For regional or comparative queries, always search for both the specific region AND the national/total figures separately.
7. Never return an empty answer. If you have retrieved relevant data, synthesize it even if partial.
8. Only answer when you are confident. State uncertainty explicitly if it exists.

Your final answer must include:
- A direct answer to the question
- All supporting citations (page + quote)
- Any calculations performed (show the expression and result)
"""


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def _sse(event: str, data: Any) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_agent(query: str) -> Generator[str, None, None]:
    """
    Streaming generator — yields SSE strings as the agent works.
    Meant to be consumed by FastAPI's StreamingResponse.
    """
    run_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    trace: list[dict] = []
    citations: list[dict] = []
    calculations: list[dict] = []

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    client = _get_client()

    yield _sse("start", {"run_id": run_id, "query": query, "model": MODEL})

    iteration = 0
    final_answer = ""

    last_text = ""  # fallback if MAX_ITERATIONS hit mid-tool-call

    while iteration < MAX_ITERATIONS:
        iteration += 1

        yield _sse("iteration", {"iteration": iteration, "status": "Calling LLM..."})
        console.print(f"\n[bold yellow]── Iteration {iteration} ──[/bold yellow]")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.1,
        )

        if not response.choices:
            console.print(f"[red]Empty response from API (rate limit or error), retrying in 5s...[/red]")
            import time as _t; _t.sleep(5)
            continue

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        step = {
            "iteration": iteration,
            "finish_reason": finish_reason,
            "tool_calls": [],
            "text_output": msg.content or "",
        }

        if msg.content:
            last_text = msg.content  # track latest reasoning text
            console.print(f"[dim]{msg.content[:300]}[/dim]")

        if finish_reason == "stop" or not msg.tool_calls:
            final_answer = msg.content or ""
            trace.append(step)
            break

        messages.append(msg)

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            yield _sse("tool_call", {"iteration": iteration, "tool": fn_name, "args": fn_args})
            console.print(f"[green]→ Tool:[/green] [bold]{fn_name}[/bold]({json.dumps(fn_args)[:120]})")

            result = execute_tool(fn_name, fn_args)

            if fn_name == "cite" and "citation" in result:
                citations.append(result["citation"])
                yield _sse("citation", result["citation"])
            if fn_name == "calculate" and "error" not in result:
                calculations.append(result)
                yield _sse("calculation", result)

            result_str = json.dumps(result, indent=2)
            console.print(f"[blue]← Result:[/blue] {result_str[:300]}")

            yield _sse("tool_result", {
                "iteration": iteration,
                "tool": fn_name,
                "result_preview": result_str[:400],
            })

            step["tool_calls"].append({"tool": fn_name, "args": fn_args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_str,
            })

        trace.append(step)

    # If we exhausted iterations without a clean stop, use last reasoning text
    if not final_answer and last_text:
        final_answer = last_text + "\n\n_(Note: agent reached max iterations; answer compiled from final reasoning step.)_"
        console.print("[yellow]Max iterations reached — using last reasoning text as answer.[/yellow]")

    elapsed = round(time.time() - start_time, 2)

    output = {
        "run_id": run_id,
        "query": query,
        "answer": final_answer,
        "citations": citations,
        "calculations": calculations,
        "iterations": iteration,
        "elapsed_seconds": elapsed,
        "model": MODEL,
        "trace": trace,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    _save_log(run_id, output)
    console.print(f"\n[bold green]Done in {elapsed}s ({iteration} iterations)[/bold green]")

    yield _sse("done", {
        "run_id": run_id,
        "query": query,
        "answer": final_answer,
        "citations": citations,
        "calculations": calculations,
        "iterations": iteration,
        "elapsed_seconds": elapsed,
        "model": MODEL,
        "trace": trace,
        "timestamp": output["timestamp"],
    })


def run_agent(query: str) -> dict[str, Any]:
    """Blocking wrapper — consumes the stream and returns the final result dict."""
    final: dict = {}
    for event_str in stream_agent(query):
        # Parse SSE to find the "done" event
        for line in event_str.strip().split("\n"):
            if line.startswith("data:"):
                pass
        if '"done"' in event_str or "event: done" in event_str:
            data_line = [l for l in event_str.split("\n") if l.startswith("data:")][0]
            final = json.loads(data_line[len("data:"):].strip())
    return final


def _save_log(run_id: str, data: dict) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"run_{run_id}.json"
    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)
    console.print(f"[dim]Trace saved to {log_path}[/dim]")
