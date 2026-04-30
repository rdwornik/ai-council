"""Research output: save markdown report to disk and print console summary."""

import logging
import re
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

from ai_council.research.models import MergedResearchReport


def _slug(query: str, max_len: int = 50) -> str:
    s = query.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    return s[:max_len].rstrip("-")


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs:02d}s"


def save_research_to_file(
    report: MergedResearchReport,
    output_dir: Path,
    from_cache: bool = False,
    secondary_dir: Path | None = None,
) -> list[Path]:
    """Save merged research report as markdown. Returns list of paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slug(report.query)
    filename = f"council_out_{ts}_research_{slug}.md"
    file_path = output_dir / filename

    lines: list[str] = [
        "# Research Report\n",
        f"**Query:** {report.query}\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if from_cache:
        lines.append(f"**Source:** cache (key: {report.cache_key})")
    lines.append(f"**Total cost:** ${report.total_cost_usd:.4f}")
    lines.append(f"**Duration:** {_format_duration(report.total_duration_sec)}")
    lines.append(f"**Sources found:** {report.total_sources}")
    lines.append("")

    # Provider summary table
    lines.append("## Provider Summary\n")
    lines.append("| Provider | Status | Duration | Cost | Sources |")
    lines.append("|----------|--------|----------|------|---------|")
    for r in report.results:
        status = "error" if r.error else ("timeout" if r.timed_out else "ok")
        dur_str = _format_duration(r.duration_sec) if r.duration_sec else "—"
        cost_str = f"${r.cost_usd:.4f}" if r.cost_usd else "—"
        src_count = str(len(r.sources)) if r.sources else "0"
        lines.append(f"| {r.provider} | {status} | {dur_str} | {cost_str} | {src_count} |")
    lines.append("")

    # Summary section
    lines.append("## Summary\n")
    lines.append(report.summary_2500)
    lines.append("")

    # Full merged report
    lines.append("---\n")
    lines.append("## Full Research Report\n")
    lines.append(report.merged_report)

    content = "\n".join(lines)
    file_path.write_text(content, encoding="utf-8")
    saved = [file_path]

    if secondary_dir is not None:
        if secondary_dir.exists():
            secondary_path = secondary_dir / filename
            secondary_path.write_text(content, encoding="utf-8")
            saved.append(secondary_path)
        else:
            logging.getLogger(__name__).warning(
                "Secondary output dir not found: %s — research report saved to primary only.",
                secondary_dir,
            )

    return saved


def print_research_summary(
    report: MergedResearchReport,
    file_path: Path | None,
    from_cache: bool,
    console: Console | None = None,
) -> None:
    """Print a Rich console summary of the research report."""
    if console is None:
        console = Console(legacy_windows=False)

    console.print(Rule("[bold cyan]Research Results[/bold cyan]"))
    console.print()
    console.print(f"[bold]Query:[/bold] {report.query}")
    if from_cache:
        console.print(f"[dim](loaded from cache — key: {report.cache_key})[/dim]")
    console.print()

    # Provider results
    successful = [r for r in report.results if not r.error and r.content]
    failed = [r for r in report.results if r.error]

    total_sources = sum(len(r.sources) for r in successful if r.sources)
    console.print(f"[bold]Providers:[/bold] {len(successful)} succeeded, {len(failed)} failed | {total_sources} sources total")
    for r in report.results:
        if r.error:
            icon = "[red]✗[/red]"
            detail = f"[red]{r.error[:80]}[/red]"
        else:
            icon = "[green]✓[/green]"
            detail = (
                f"[dim]{_format_duration(r.duration_sec)}[/dim]"
                f"  [dim]${r.cost_usd:.4f}[/dim]"
                f"  [dim]{len(r.sources)} sources[/dim]"
            )
        console.print(f"  {icon} [bold]{r.provider:<16}[/bold] {detail}")

    console.print()
    console.print(
        f"[dim]Total: {_format_duration(report.total_duration_sec)}"
        f"  |  ${report.total_cost_usd:.4f}"
        f"  |  {report.total_sources} unique sources[/dim]"
    )
    console.print()

    # Print summary section
    console.print(Rule("[bold]Summary[/bold]"))
    console.print()
    console.print(report.summary_2500)
    console.print()

    if file_path:
        console.print(Rule())
        console.print(f"[dim]Saved: {file_path}[/dim]")
