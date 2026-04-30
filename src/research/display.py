"""Progressive research display using Rich Live.

Runs all providers in parallel via asyncio.as_completed() and updates
a live status table as each provider completes or fails.
"""

import asyncio
import logging
import time
from typing import Sequence

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from src.research.models import ResearchResult
from src.research.provider import ResearchProvider, ResearchProviderError

logger = logging.getLogger(__name__)

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs:02d}s"


def _make_status_table(
    provider_names: list[str],
    statuses: dict[str, str],       # "waiting" | "running" | "done" | "error" | "timeout"
    results: dict[str, ResearchResult | None],
    elapsed: dict[str, float],
    spinner_frame: int,
) -> Table:
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("Provider", style="bold", min_width=22)
    table.add_column("Status", min_width=14)
    table.add_column("Time", justify="right", min_width=8)
    table.add_column("Sources", justify="right", min_width=7)
    table.add_column("Cost", justify="right", min_width=8)

    spinner = _SPINNER_FRAMES[spinner_frame % len(_SPINNER_FRAMES)]

    for name in provider_names:
        status = statuses.get(name, "waiting")
        result = results.get(name)
        dur = elapsed.get(name, 0.0)

        if status == "waiting":
            status_text = Text("waiting", style="dim")
            time_str = "—"
            cost_str = "—"
            src_str = "—"
        elif status == "running":
            status_text = Text(f"{spinner} running", style="yellow")
            time_str = _format_duration(dur)
            cost_str = "—"
            src_str = "—"
        elif status == "done" and result is not None:
            status_text = Text("✓ done", style="green")
            time_str = _format_duration(dur)
            cost_str = f"${result.cost_usd:.3f}" if result.cost_usd else "—"
            src_str = str(len(result.sources)) if result.sources else "0"
        elif status == "timeout":
            status_text = Text("✗ timeout", style="red")
            time_str = _format_duration(dur)
            reason = result.error if result and result.error else "timed out"
            cost_str = f"({reason[:40]})" if len(reason) <= 40 else f"({reason[:39]}…)"
            src_str = "—"
        else:  # error
            status_text = Text("✗ failed", style="red")
            time_str = _format_duration(dur)
            reason = result.error if result and result.error else "unknown error"
            cost_str = f"({reason[:40]})" if len(reason) <= 40 else f"({reason[:39]}…)"
            src_str = "—"

        table.add_row(name, status_text, time_str, src_str, cost_str)

    return table


async def run_research_with_display(
    providers: Sequence[ResearchProvider],
    query: str,
    console: Console | None = None,
) -> list[ResearchResult]:
    """Run all providers in parallel; display live progress; return all results.

    Results include failed/timed-out providers with error field set.
    Order of returned list matches input providers order.
    """
    if console is None:
        console = Console()

    provider_names = [p.name() for p in providers]
    statuses: dict[str, str] = {n: "waiting" for n in provider_names}
    results: dict[str, ResearchResult | None] = {n: None for n in provider_names}
    start_times: dict[str, float] = {}
    elapsed: dict[str, float] = {n: 0.0 for n in provider_names}

    console.print(f"\n[bold cyan]Research:[/bold cyan] {query}\n")

    async def _run_one(provider: ResearchProvider) -> tuple[str, ResearchResult | None]:
        name = provider.name()
        statuses[name] = "running"
        start_times[name] = time.monotonic()
        try:
            result = await provider.research(query)
            elapsed[name] = time.monotonic() - start_times[name]
            statuses[name] = "done"
            return name, result
        except ResearchProviderError as exc:
            elapsed[name] = time.monotonic() - start_times[name]
            statuses[name] = "timeout" if "Timed out" in str(exc) else "error"
            logger.warning("Research provider %s failed: %s", name, exc)
            error_result = _error_result(provider, query, str(exc))
            return name, error_result
        except Exception as exc:
            elapsed[name] = time.monotonic() - start_times[name]
            statuses[name] = "error"
            logger.warning("Research provider %s unexpected error: %s", name, exc)
            error_result = _error_result(provider, query, str(exc))
            return name, error_result

    tasks = [asyncio.create_task(_run_one(p)) for p in providers]
    frame = 0

    with Live(console=console, refresh_per_second=4) as live:
        pending = set(tasks)
        done_tasks: set[asyncio.Task] = set()

        while pending:
            # Update elapsed for running providers
            now = time.monotonic()
            for name in provider_names:
                if statuses[name] == "running" and name in start_times:
                    elapsed[name] = now - start_times[name]

            live.update(_make_status_table(provider_names, statuses, results, elapsed, frame))
            frame += 1

            # Wait for next completion (short timeout to keep spinner alive)
            done_now, pending = await asyncio.wait(pending, timeout=0.25, return_when=asyncio.FIRST_COMPLETED)
            done_tasks |= done_now

            for task in done_now:
                name, result = await task
                results[name] = result

        # Final render (all done)
        now = time.monotonic()
        for name in provider_names:
            if statuses[name] == "running" and name in start_times:
                elapsed[name] = now - start_times[name]
        live.update(_make_status_table(provider_names, statuses, results, elapsed, frame))

    console.print()

    # Return in original order
    return [results[p.name()] for p in providers]  # type: ignore[misc]


def _error_result(provider: ResearchProvider, query: str, error: str) -> ResearchResult:
    from src.research.models import ResearchResult
    return ResearchResult(
        provider=provider.name(),
        query=query,
        content="",
        error=error,
        timed_out="Timed out" in error,
    )
