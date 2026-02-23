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

console = Console(legacy_windows=False)


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
        preview += "..."
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
    synth_label = result.synthesizer
    if result.synthesizer_is_participant:
        synth_label += " (participant)"
    else:
        synth_label += " (non-participant)"
    console.print(
        Text(
            f"Synthesized by: {synth_label} | "
            f"Duration: {result.total_duration_sec:.1f}s | "
            f"Rounds: {len(result.rounds)} | "
            f"Mode: {result.panel_mode}",
            style="dim",
        )
    )
    console.print(Markdown(result.synthesis))


def save_to_file(result: DebateResult, output_dir: Path, slug_override: str | None = None) -> Path:
    """Save the full debate transcript as a markdown file.

    Args:
        result: The completed DebateResult.
        output_dir: Directory to save the file in.
        slug_override: If provided, use this as the filename stem instead of
            deriving one from the question text. Useful for inbox mode.

    Returns:
        Path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slug_override if slug_override is not None else _slug(result.question.text)
    filename = f"{timestamp}_{slug}.md"
    filepath = output_dir / filename

    # Derive panel info from first round responses
    panel_providers = sorted({r.provider for r in result.rounds[0].responses})
    panel_models = [
        next(r.model for r in result.rounds[0].responses if r.provider == p)
        for p in panel_providers
    ]
    panel_str = ", ".join(panel_models)

    synth_model = next(
        (r.model for rnd in result.rounds for r in rnd.responses if r.provider == result.synthesizer),
        result.synthesizer,
    )
    synth_label = synth_model
    if result.synthesizer_is_participant:
        synth_label += " (participant)"
    else:
        synth_label += " (non-participant)"

    panel_count = len(panel_providers)
    if result.panel_mode == "default":
        mode_str = f"default ({panel_count}-model panel)"
    elif result.panel_mode == "full":
        mode_str = f"full ({panel_count}-model panel)"
    else:
        mode_str = "custom"

    lines: list[str] = [
        f"# AI Council Debate: {result.question.text[:80]}",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Panel:** {panel_str}",
        f"**Synthesizer:** {synth_label}",
        f"**Rounds:** {len(result.rounds)}",
        f"**Duration:** {result.total_duration_sec:.1f}s",
        f"**Mode:** {mode_str}",
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

    synth_is_label = "participant" if result.synthesizer_is_participant else "non-participant"
    lines += [
        f"## Synthesis (by {result.synthesizer}, {synth_is_label})",
        "",
        result.synthesis,
        "",
    ]

    filepath.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Debate saved to: %s", filepath)
    return filepath
