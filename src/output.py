"""Rich console output and markdown file save for debate results."""

import json
import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.tree import Tree

from src.models import DebateMetrics, DebateResult, ModelResponse

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
    if result.degraded:
        console.print(
            Panel(
                f"[bold yellow]DEGRADED RUN[/bold yellow]\n"
                f"{result.degradation_summary or 'Some providers failed during the debate.'}",
                border_style="yellow",
            )
        )
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


def print_cost_summary(metrics: DebateMetrics) -> None:
    """Print a cost breakdown tree to the console."""
    tree = Tree("[bold]Cost Summary[/bold]")

    # Group debate calls by round
    by_round: dict[int, list] = {}
    synthesis_call = None
    for call in metrics.calls:
        if call.round_number == 0:
            synthesis_call = call
        else:
            by_round.setdefault(call.round_number, []).append(call)

    for rnd_num in sorted(by_round):
        calls = by_round[rnd_num]
        rnd_tokens = sum(c.input_tokens + c.output_tokens for c in calls)
        rnd_cost = sum(c.estimated_cost_usd for c in calls)
        tree.add(
            f"Round {rnd_num}: [green]${rnd_cost:.4f}[/green]"
            f" ({len(calls)} providers, {rnd_tokens:,} tokens)"
        )

    if synthesis_call:
        synth_tokens = synthesis_call.input_tokens + synthesis_call.output_tokens
        tree.add(
            f"Synthesis: [green]${synthesis_call.estimated_cost_usd:.4f}[/green]"
            f" (1 provider, {synth_tokens:,} tokens)"
        )

    total_tokens = metrics.total_input_tokens + metrics.total_output_tokens
    tree.add(
        f"[bold]Total: [green]${metrics.total_estimated_cost_usd:.4f}[/green][/bold]"
        f" ({total_tokens:,} tokens, {metrics.total_duration_sec:.1f}s)"
    )

    console.print(Panel(tree, border_style="dim"))


def save_to_file(
    result: DebateResult,
    output_dir: Path,
    slug_override: str | None = None,
    secondary_dir: Path | None = None,
) -> list[Path]:
    """Save the full debate transcript as a markdown file.

    Writes to output_dir (always) and secondary_dir (if it exists on disk).

    Returns:
        List of paths written. First entry is always the primary path.
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
        (
            r.model
            for rnd in result.rounds
            for r in rnd.responses
            if r.provider == result.synthesizer
        ),
        result.synthesizer,
    )
    synth_label = synth_model
    if result.synthesizer_is_participant:
        synth_label += " (participant)"
    else:
        synth_label += " (non-participant)"

    panel_count = len(panel_providers)
    if result.panel_mode == "default":
        panel_mode_str = f"default ({panel_count}-model panel)"
    elif result.panel_mode == "full":
        panel_mode_str = f"full ({panel_count}-model panel)"
    else:
        panel_mode_str = "custom"

    cost_line = ""
    if result.metrics:
        total_tokens = result.metrics.total_input_tokens + result.metrics.total_output_tokens
        cost_line = f"**Cost:** ~${result.metrics.total_estimated_cost_usd:.4f} ({total_tokens:,} tokens)"

    lines: list[str] = [
        f"# AI Council Debate: {result.question.text[:80]}",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Panel:** {panel_str}",
        f"**Synthesizer:** {synth_label}",
        f"**Rounds:** {len(result.rounds)}",
        f"**Duration:** {result.total_duration_sec:.1f}s",
        f"**Panel Mode:** {panel_mode_str}",
        f"**Debate Mode:** {result.mode}",
        f"**Source:** {result.question.source}",
    ]
    if cost_line:
        lines.append(cost_line)
    if result.degraded:
        failed = [k for k, v in result.provider_statuses.items() if v == "failed"]
        degradation_note = result.degradation_summary or "Some providers failed during the debate."
        lines.append(f"**Status:** DEGRADED — {degradation_note}")
        if failed:
            lines.append(f"**Failed providers:** {', '.join(failed)}")

    # Provider notes: retried-and-recovered + skipped providers
    retried = sorted({
        r.provider
        for rnd in result.rounds
        for r in rnd.responses
        if r.was_retry
    })
    failed_providers = [k for k, v in result.provider_statuses.items() if v == "failed"]
    provider_note_parts: list[str] = []
    for p in retried:
        provider_note_parts.append(f"{p} retried (timeout, recovered)")
    for p in failed_providers:
        provider_note_parts.append(f"{p} skipped")
    if provider_note_parts:
        lines.append(f"**Provider Notes:** {'; '.join(provider_note_parts)}.")
    lines += [
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

    synth_is_label = (
        "participant" if result.synthesizer_is_participant else "non-participant"
    )
    lines += [
        f"## Synthesis (by {result.synthesizer}, {synth_is_label})",
        "",
        result.synthesis,
        "",
    ]

    filepath.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Debate saved to: %s", filepath)
    saved = [filepath]

    if result.metrics:
        _save_metrics_json(result, filepath)

    if secondary_dir is not None:
        if secondary_dir.exists():
            secondary_path = secondary_dir / filename
            secondary_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("Transcript copied to: %s", secondary_path)
            saved.append(secondary_path)
            if result.metrics:
                _save_metrics_json(result, secondary_path)
        else:
            logger.warning(
                "Secondary output dir not found: %s — transcript saved to primary only.",
                secondary_dir,
            )

    return saved


def _save_metrics_json(result: DebateResult, transcript_path: Path) -> None:
    """Save detailed metrics as a JSON file alongside the transcript."""
    assert result.metrics is not None
    m = result.metrics
    payload = {
        "debate_id": transcript_path.stem,
        "total_input_tokens": m.total_input_tokens,
        "total_output_tokens": m.total_output_tokens,
        "total_tokens": m.total_input_tokens + m.total_output_tokens,
        "total_estimated_cost_usd": round(m.total_estimated_cost_usd, 6),
        "total_duration_sec": round(m.total_duration_sec, 2),
        "calls": [
            {
                "provider": c.provider,
                "round_number": c.round_number,
                "input_tokens": c.input_tokens,
                "output_tokens": c.output_tokens,
                "estimated_cost_usd": round(c.estimated_cost_usd, 6),
                "latency_sec": round(c.latency_sec, 3),
                "was_retry": c.was_retry,
            }
            for c in m.calls
        ],
    }
    metrics_path = transcript_path.with_suffix("").with_name(
        transcript_path.stem + "_metrics.json"
    )
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Metrics saved to: %s", metrics_path)
