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

from ai_council.models import DebateMetrics, DebateResult, ModelResponse

logger = logging.getLogger(__name__)

console = Console(legacy_windows=False)


def _slug(text: str, max_len: int = 50) -> str:
    """Convert text to a filename-safe slug (hyphens, no special chars)."""
    import re

    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:max_len].rstrip("-")


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


def _write_routed(
    content: str,
    filename: str,
    output_dir: Path,
    secondary_dir: Path | None,
    target_paths: list[Path] | None,
    return_dir: Path | None,
) -> list[Path]:
    """Write `content` as `filename` to the canonical dir plus any optional routes.

    The canonical `output_dir` is ALWAYS written first (ADR-10 / ADR-43: the return
    is a copy/route, never a replacement). Then, in order:
      - `secondary_dir`: legacy mirror, written only if it already exists on disk;
      - `return_dir`: ADR-10 deterministic return (auto-mkdir, best-effort);
      - each `target_paths` dir: ADR-43 per-invocation mirror (auto-mkdir, best-effort).

    Returns the list of paths written, canonical first.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = output_dir / filename
    primary.write_text(content, encoding="utf-8")
    saved = [primary]

    if secondary_dir is not None:
        if secondary_dir.exists():
            secondary_path = secondary_dir / filename
            secondary_path.write_text(content, encoding="utf-8")
            logger.info("Copied to: %s", secondary_path)
            saved.append(secondary_path)
        else:
            logger.warning(
                "Secondary output dir not found: %s — saved to primary only.",
                secondary_dir,
            )

    if return_dir is not None:
        try:
            return_dir.mkdir(parents=True, exist_ok=True)
            return_path = return_dir / filename
            return_path.write_text(content, encoding="utf-8")
            logger.info("Deterministic return written to: %s", return_path)
            saved.append(return_path)
        except Exception as exc:
            logger.warning("Return-dir write failed for %s: %s", return_dir, exc)

    for target_dir in target_paths or []:
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / filename
            target_path.write_text(content, encoding="utf-8")
            logger.info("Mirrored to: %s", target_path)
            saved.append(target_path)
        except Exception as exc:
            logger.warning("Mirror write failed for %s: %s", target_dir, exc)

    return saved


def save_to_file(
    result: DebateResult,
    output_dir: Path,
    slug_override: str | None = None,
    secondary_dir: Path | None = None,
    target_paths: list[Path] | None = None,
    return_dir: Path | None = None,
) -> list[Path]:
    """Save the full debate transcript as a markdown file.

    Writes to output_dir (always, canonical), secondary_dir (if it exists on disk),
    return_dir (ADR-10 deterministic return; auto-mkdir, best-effort), and each path
    in target_paths (auto-mkdir, best-effort).

    Returns:
        List of paths written. First entry is always the primary path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slug_override if slug_override is not None else _slug(result.question.text)
    filename = f"council-out-{timestamp}-{result.mode}-{slug}.md"

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
        "## Question",
        "",
        result.question.text,
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

    saved = _write_routed(
        "\n".join(lines), filename, output_dir, secondary_dir, target_paths, return_dir
    )
    logger.info("Debate saved to: %s", saved[0])

    if result.metrics:
        _save_metrics_json(result, saved[0])

    return saved


# Headings whose section carries dissent in the synthesizer's structured verdict.
# "Unresolved Disagreements" (pick), "Contested Points" (judge), or explicit dissent.
_DISSENT_HEADING_MARKERS = (
    "unresolved disagreement",
    "contested point",
    "dissent",
    "minority",
)

# A dissent section body is treated as "no genuine dissent" when it opens with one of
# these negations (the synthesizer reported consensus rather than a split).
_NO_DISSENT_PREFIXES = (
    "none",
    "n/a",
    "no disagreement",
    "no unresolved",
    "no contested",
    "no dissent",
    "no minority",
    "there were no",
    "there was no",
    "the panel reached consensus",
    "the panel agreed",
    "consensus",
    "unanimous",
)


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs on level-2 ('## ') headings."""
    sections: list[tuple[str, str]] = []
    heading: str | None = None
    body: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if heading is not None:
                sections.append((heading, "\n".join(body)))
            heading = line[3:].strip()
            body = []
        elif heading is not None:
            body.append(line)
    if heading is not None:
        sections.append((heading, "\n".join(body)))
    return sections


def _is_genuine_dissent(body: str) -> bool:
    """True when a dissent section body has real content (not a 'none/consensus' note)."""
    stripped = body.strip()
    if not stripped:
        return False
    first_line = ""
    for ln in stripped.splitlines():
        t = ln.strip().lstrip("-*# ").strip()
        if t:
            first_line = t.lower()
            break
    if not first_line or any(first_line.startswith(p) for p in _NO_DISSENT_PREFIXES):
        return False
    return len(stripped) >= 12


def extract_dissent(synthesis: str) -> str | None:
    """Return formatted dissent markdown when the verdict is non-unanimous, else None.

    ai-council has no structured vote tally — ADR-03 blind voting is free-text critique
    and the verdict is the synthesizer's narrative. The machine-available signal of a
    non-unanimous outcome is therefore a *substantive* disagreement/dissent section in
    that verdict. This surfaces exactly what Rama 4 / audit G7 flag (dissent buried in
    the synthesis) without changing any Council runtime behavior.
    """
    kept: list[tuple[str, str]] = []
    for heading, body in _split_sections(synthesis):
        h = heading.lower()
        if any(marker in h for marker in _DISSENT_HEADING_MARKERS) and _is_genuine_dissent(body):
            kept.append((heading, body.strip()))
    if not kept:
        return None
    parts: list[str] = []
    for heading, body in kept:
        parts.extend([f"## {heading}", "", body, ""])
    return "\n".join(parts).strip()


def save_minority_report(
    result: DebateResult,
    output_dir: Path,
    slug_override: str | None = None,
    secondary_dir: Path | None = None,
    target_paths: list[Path] | None = None,
    return_dir: Path | None = None,
) -> list[Path]:
    """Emit the minority/dissent report as a discrete, durable artifact (Rama 4, #15).

    Fires only when the verdict is non-unanimous (see extract_dissent). Routed to the
    same destinations as the verdict via _write_routed (canonical + secondary + return +
    targets), so a --return-dir also receives it. Returns [] when there is no dissent.
    """
    body = extract_dissent(result.synthesis)
    if body is None:
        return []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slug_override if slug_override is not None else _slug(result.question.text)
    filename = f"council-minority-{timestamp}-{result.mode}-{slug}.md"

    synth_is_label = "participant" if result.synthesizer_is_participant else "non-participant"
    lines = [
        f"# AI Council Minority Report: {result.question.text[:80]}",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Debate Mode:** {result.mode}",
        f"**Synthesizer:** {result.synthesizer} ({synth_is_label})",
        f"**Source:** {result.question.source}",
        "",
        "> First-class dissent artifact (Rama 4). The verdict was NOT unanimous: the",
        "> synthesizer recorded unresolved disagreement. The dissenting positions are",
        "> preserved below so they are not lost inside the synthesized verdict.",
        "",
        "---",
        "",
        body,
        "",
    ]
    saved = _write_routed(
        "\n".join(lines), filename, output_dir, secondary_dir, target_paths, return_dir
    )
    logger.info("Minority report saved to: %s", saved[0])
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
    # Sidecar extension mechanism (this arc is the first lander — L-CLI seam / ADR-12).
    # The _metrics.json sidecar is extended by NAMESPACED, ADDITIVE top-level keys, one
    # namespace owned by one lane: `seats` (L-CLI, per-seat backend/identity/fallback) and
    # `synthesis` (L-EPI, below) — neither nests inside `calls`, neither extends the other.
    # Consumers never branch on backend: every seat gets a uniform entry (API seats carry
    # identity_channel="api-echo"). Model/seat/key values are names only, never secrets.
    if m.seats:
        payload["seats"] = [
            {
                "seat": seat.seat,
                "requested_backend": seat.requested_backend,
                "actual_backend": seat.actual_backend,
                "cli": seat.cli,
                "requested_model": seat.requested_model,
                "actual_model": seat.actual_model,
                "identity_channel": seat.identity_channel,
                "identity_readable": seat.identity_readable,
                "fallback_events": [
                    {
                        "round": fe.round,
                        "from_backend": fe.from_backend,
                        "to_backend": fe.to_backend,
                        "cause": fe.cause,
                        "detail": fe.detail,
                    }
                    for fe in seat.fallback_events
                ],
            }
            for seat in m.seats
        ]
    if result.synthesis_metrics is not None:
        s = result.synthesis_metrics
        payload["synthesis"] = {
            "synthesizer_model": s.synthesizer_model,
            "transcript_size_tokens": s.transcript_size_tokens,
            "output_tokens": s.output_tokens,
            "synth_latency_seconds": round(s.synth_latency_seconds, 3),
            "error_class": s.error_class,
        }
    metrics_path = transcript_path.with_suffix("").with_name(
        transcript_path.stem + "_metrics.json"
    )
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Metrics saved to: %s", metrics_path)
