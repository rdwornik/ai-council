"""Research output: save markdown report to disk and print console summary."""

import re
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

from ai_council.output import _write_routed
from ai_council.research.models import MergedResearchReport

# #42: the research filename already carries the `research-` mode token, so a slug
# from a query that itself begins "research…" would double it (council-out-…-research-
# research-…). Strip a leading "research" token from the slug — mirrors the leading-
# "council" strip in inbox.clean_slug. Fires only when a separator + content follow, so
# "research" alone and words like "researcher" are preserved.
_LEADING_RESEARCH_RE = re.compile(r"^research[-_ ]+(.+)$", re.IGNORECASE)


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
    target_paths: list[Path] | None = None,
    return_dir: Path | None = None,
) -> list[Path]:
    """Save merged research report as markdown. Returns list of paths written.

    Routing (canonical always first, then secondary / return-dir / target copies)
    is delegated to the shared debate-path writer so research honors --return-dir
    identically to the debate path (#23).
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slug(report.query)
    research_m = _LEADING_RESEARCH_RE.match(slug)
    if research_m:
        slug = research_m.group(1)
    filename = f"council-out-{ts}-research-{slug}.md"

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

    if report.degraded:
        successful_count = sum(1 for r in report.results if not r.error and r.content)
        total_panel = successful_count + report.failed_count
        lines.append(
            f"> [!WARNING] **Degraded research panel** — {report.failed_count} of "
            f"{total_panel} providers failed (includes build-time drops for missing API keys). "
            "Report below is based on the survivors only. See ADR-08."
        )
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
    return _write_routed(
        content,
        filename,
        output_dir,
        secondary_dir,
        target_paths,
        return_dir,
        artifact="research report",
        return_dir_required=False,
    )


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
    console.print(f"[bold]Providers:[/bold] {len(successful)} succeeded, {len(failed)} failed | {total_sources} sources total")  # noqa: E501
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

    if report.degraded:
        successful_count = sum(1 for r in report.results if not r.error and r.content)
        total_panel = successful_count + report.failed_count
        console.print(Rule("[bold red]!! DEGRADED RESEARCH PANEL !![/bold red]", style="red"))
        console.print(
            f"[bold red]{report.failed_count} of {total_panel} providers failed[/bold red] "
            "(includes build-time drops for missing API keys). "
            f"Report based on {successful_count} survivor(s). "
            "[dim]See ADR-08; process will exit with code 3.[/dim]"
        )
        failed_names = [r.provider for r in report.results if r.error]
        if failed_names:
            console.print(f"[dim red]Failed providers (API-call): {', '.join(failed_names)}[/dim red]")
        console.print()

    # Print summary section
    console.print(Rule("[bold]Summary[/bold]"))
    console.print()
    console.print(report.summary_2500)
    console.print()

    if file_path:
        console.print(Rule())
        console.print(f"[dim]Saved: {file_path}[/dim]")
