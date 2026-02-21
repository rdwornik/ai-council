"""Rich console output and markdown file save for debate results."""

import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from src.models import DebateResult, ModelResponse

logger = logging.getLogger(__name__)

console = Console()


def _slug(text: str, max_len: int = 40) -> str:
    """Convert text to a filename-safe slug."""
    import re
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:max_len]


def _response_preview(response: ModelResponse, words: int = 50) -> str:
    """Return first N words of a response."""
    all_words = response.content.split()
    preview = " ".join(all_words[:words])
    if len(all_words) > words:
        preview += "…"
    return preview


def print_round_summary(round_num: int, responses: list[ModelResponse]) -> None:
    """Print a brief summary of round responses to the console."""
    console.print(Rule(f"[bold cyan]Round {round_num} Summary[/bold cyan]"))
    for resp in responses:
        preview = _response_preview(resp)
        console.print(
            Panel(
                preview,
                title=f"[bold]{resp.provider}[/bold] ({resp.model})",
                subtitle=f"{resp.latency_sec:.1f}s",
                border_style="dim",
            )
        )


def print_synthesis(result: DebateResult) -> None:
    """Print the full synthesis to the console using Rich markdown."""
    console.print(Rule("[bold green]Council Synthesis[/bold green]"))
    console.print(
        Text(
            f"Synthesized by: {result.synthesizer} | "
            f"Duration: {result.total_duration_sec:.1f}s | "
            f"Rounds: {len(result.rounds)}",
            style="dim",
        )
    )
    console.print(Markdown(result.synthesis))


def save_to_file(result: DebateResult, output_dir: Path) -> Path:
    """Save the full debate transcript as a markdown file.

    Args:
        result: The completed DebateResult.
        output_dir: Directory to save the file in.

    Returns:
        Path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slug(result.question.text)
    filename = f"{timestamp}_{slug}.md"
    filepath = output_dir / filename

    model_list = ", ".join(
        sorted({r.provider for rnd in result.rounds for r in rnd.responses})
    )

    lines: list[str] = [
        f"# AI Council Debate: {result.question.text[:50]}",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Models:** {model_list}",
        f"**Rounds:** {len(result.rounds)}",
        f"**Duration:** {result.total_duration_sec:.1f}s",
        f"**Source:** {result.question.source}",
        "",
        "---",
        "",
    ]

    for rnd in result.rounds:
        round_label = "Initial Responses" if rnd.number == 1 else "Critique"
        lines.append(f"## Round {rnd.number}: {round_label}")
        lines.append("")
        for resp in rnd.responses:
            lines.append(f"### {resp.provider.title()} ({resp.model})")
            lines.append("")
            lines.append(resp.content)
            lines.append("")
            lines.append(
                f"*Latency: {resp.latency_sec:.2f}s"
                + (f" | Tokens: {resp.token_count}" if resp.token_count else "")
                + "*"
            )
            lines.append("")

    lines += [
        f"## Synthesis (by {result.synthesizer})",
        "",
        result.synthesis,
        "",
    ]

    filepath.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Debate saved to: %s", filepath)
    return filepath
