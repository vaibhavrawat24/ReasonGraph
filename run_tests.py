"""
Runs the three evaluation test queries and saves logs.

Traces are written to two places:
  logs/            — gitignored, full runtime logs
  sample_logs/     — committed to git, named per test for grader review
"""

import json
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.rule import Rule

load_dotenv()

from agent.orchestrator import run_agent

console = Console()

SAMPLE_LOGS_DIR = Path("sample_logs")

TEST_QUERIES = [
    {
        "id": "test_1_verification",
        "query": "What is the total number of jobs reported, and where exactly is this stated?",
    },
    {
        "id": "test_2_synthesis",
        "query": "Compare the concentration of 'Pure-Play' cybersecurity firms in the South-West against the National Average.",
    },
    {
        "id": "test_3_forecasting",
        "query": "Based on our 2022 baseline and the stated 2030 job target, what is the required compound annual growth rate (CAGR) to hit that goal?",
    },
]


def main():
    console.print(Rule("[bold cyan]ReasonGraph — Evaluation Test Suite[/bold cyan]"))
    SAMPLE_LOGS_DIR.mkdir(exist_ok=True)

    results = []
    for test in TEST_QUERIES:
        console.print(Rule(f"[yellow]{test['id']}[/yellow]"))
        result = run_agent(test["query"])
        result["test_id"] = test["id"]
        results.append(result)

        console.print(f"\n[bold green]ANSWER:[/bold green]\n{result['answer']}\n")

        if result["citations"]:
            console.print("[bold]Citations:[/bold]")
            for c in result["citations"]:
                console.print(f"  Page {c['page']}: \"{c['quote'][:100]}...\"")

        if result["calculations"]:
            console.print("[bold]Calculations:[/bold]")
            for c in result["calculations"]:
                console.print(f"  {c['expression']} = {c['result_formatted']}")

        # Save individual trace to sample_logs/ (committed to git)
        sample_path = SAMPLE_LOGS_DIR / f"{test['id']}.json"
        with open(sample_path, "w") as f:
            json.dump(result, f, indent=2)
        console.print(f"[dim]Saved to {sample_path}[/dim]")

    Path("logs").mkdir(exist_ok=True)
    with open("logs/test_suite_results.json", "w") as f:
        json.dump(results, f, indent=2)

    console.print(Rule("[bold green]All tests complete. Traces in sample_logs/[/bold green]"))


if __name__ == "__main__":
    main()
