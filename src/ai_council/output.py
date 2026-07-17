"""Rich console output and markdown file save for debate results."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.tree import Tree

from ai_council.models import (
    DebateMetrics,
    DebateResult,
    FallbackEvent,
    ModelResponse,
    SeatMetrics,
)

logger = logging.getLogger(__name__)

console = Console(legacy_windows=False)


def _ts(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Single source for wall-clock timestamp strings (audit B3).

    Local-time by design: these are filename/header display strings, never compared,
    and every existing ``output/`` artifact uses local ``%Y%m%d_%H%M%S``. The machine-
    readable, tz-aware timestamp for callers lives in the verdict package's ``timestamp``
    field (``_iso_now``), not here. One place so a run's ``<ts>`` is derived consistently.
    """
    return datetime.now().strftime(fmt)


def _iso_now() -> str:
    """Tz-aware ISO-8601 timestamp for the machine-readable verdict-package field."""
    return datetime.now(timezone.utc).isoformat()


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


def _route_dirs(
    output_dir: Path,
    secondary_dir: Path | None,
    target_paths: list[Path] | None,
    return_dir: Path | None,
) -> list[Path]:
    """The directories `_write_routed` will target, in order (canonical first).

    Mirrors `_write_routed`'s destination selection so a manifest can name every planned
    path before the write; `secondary_dir` is included only when it exists (as the write is).
    """
    dirs = [output_dir]
    if secondary_dir is not None and secondary_dir.exists():
        dirs.append(secondary_dir)
    if return_dir is not None:
        dirs.append(return_dir)
    dirs.extend(target_paths or [])
    return dirs


class OutputRoutingError(RuntimeError):
    """A required output destination (e.g. an explicit --return-dir) was not written.

    Raised for the caller's *required* deliverable so a routing failure surfaces loudly
    instead of yielding a bare exit 0 (DRAFT-INT-1 R4). Optional mirrors stay best-effort.
    """


def _fallback_payload(fe: FallbackEvent) -> dict:
    """The single canonical serialization of one CLI fallback event (never a raw credential)."""
    return {
        "round": fe.round,
        "from_backend": fe.from_backend,
        "to_backend": fe.to_backend,
        "cause": fe.cause,
        "detail": fe.detail,
    }


def _seat_payload(seat: SeatMetrics) -> dict:
    """The single canonical per-seat serialization (L-CLI seats[] namespace).

    One definition, consumed by both the `_metrics.json` sidecar and the verdict package's
    `panel.seats`, so the two can never drift. Names only, never secrets.
    """
    return {
        "seat": seat.seat,
        "requested_backend": seat.requested_backend,
        "actual_backend": seat.actual_backend,
        "cli": seat.cli,
        "requested_model": seat.requested_model,
        "actual_model": seat.actual_model,
        "identity_channel": seat.identity_channel,
        "identity_readable": seat.identity_readable,
        "fallback_events": [_fallback_payload(fe) for fe in seat.fallback_events],
    }


def _panel_str(result: DebateResult) -> str:
    """The comma-joined panel model list, derived from first-round responses."""
    panel_providers = sorted({r.provider for r in result.rounds[0].responses})
    panel_models = [
        next(r.model for r in result.rounds[0].responses if r.provider == p)
        for p in panel_providers
    ]
    return ", ".join(panel_models)


def _synth_label(result: DebateResult) -> str:
    """The synthesizer's model string + (participant|non-participant) suffix."""
    synth_model = next(
        (
            r.model
            for rnd in result.rounds
            for r in rnd.responses
            if r.provider == result.synthesizer
        ),
        result.synthesizer,
    )
    suffix = " (participant)" if result.synthesizer_is_participant else " (non-participant)"
    return synth_model + suffix


def _build_header(result: DebateResult) -> list[str]:
    """The **Panel/Synthesizer/Rounds/Cost/Status/Provider-Notes** metadata lines.

    Returns only the ``**key:** value`` lines (no title, no separator) so the caller
    controls document framing. Was inlined in ``save_to_file`` (audit A4).
    """
    panel_count = len(sorted({r.provider for r in result.rounds[0].responses}))
    if result.panel_mode == "default":
        panel_mode_str = f"default ({panel_count}-model panel)"
    elif result.panel_mode == "full":
        panel_mode_str = f"full ({panel_count}-model panel)"
    else:
        panel_mode_str = "custom"

    lines: list[str] = [
        f"**Date:** {_ts('%Y-%m-%d %H:%M:%S')}",
        f"**Panel:** {_panel_str(result)}",
        f"**Synthesizer:** {_synth_label(result)}",
        f"**Rounds:** {len(result.rounds)}",
        f"**Duration:** {result.total_duration_sec:.1f}s",
        f"**Panel Mode:** {panel_mode_str}",
        f"**Debate Mode:** {result.mode}",
        f"**Source:** {result.question.source}",
    ]
    if result.metrics:
        total_tokens = result.metrics.total_input_tokens + result.metrics.total_output_tokens
        lines.append(
            f"**Cost:** ~${result.metrics.total_estimated_cost_usd:.4f} ({total_tokens:,} tokens)"
        )
    if result.degraded:
        failed = [k for k, v in result.provider_statuses.items() if v == "failed"]
        degradation_note = result.degradation_summary or "Some providers failed during the debate."
        lines.append(f"**Status:** DEGRADED — {degradation_note}")
        if failed:
            lines.append(f"**Failed providers:** {', '.join(failed)}")

    # Provider notes: retried-and-recovered + skipped providers
    retried = sorted({r.provider for rnd in result.rounds for r in rnd.responses if r.was_retry})
    failed_providers = [k for k, v in result.provider_statuses.items() if v == "failed"]
    provider_note_parts: list[str] = []
    for p in retried:
        provider_note_parts.append(f"{p} retried (timeout, recovered)")
    for p in failed_providers:
        provider_note_parts.append(f"{p} skipped")
    if provider_note_parts:
        lines.append(f"**Provider Notes:** {'; '.join(provider_note_parts)}.")
    # Human-readable mirror of the verdict package (DRAFT-INT-1, folded here per the
    # architect ruling): keeps a Lane B/operator read self-contained. Never echoes the
    # question text (the body's ## Question stays the single full-question site).
    lines.append("")
    lines.extend(_verdict_summary_lines(result))
    return lines


def _build_body(result: DebateResult) -> list[str]:
    """The ``## Question`` block, per-round transcript, and ``## Synthesis`` block.

    Everything below the ``---`` separator. Was inlined in ``save_to_file`` (audit A4).
    """
    lines: list[str] = ["## Question", "", result.question.text, ""]
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
    return lines


def save_to_file(
    result: DebateResult,
    output_dir: Path,
    slug_override: str | None = None,
    secondary_dir: Path | None = None,
    target_paths: list[Path] | None = None,
    return_dir: Path | None = None,
) -> list[Path]:
    """Save the full debate transcript as a markdown file.

    Pure orchestration (audit A4): filename derivation, content assembly via
    ``_build_header``/``_build_body``, routed write, metrics trigger. Writes to
    output_dir (always, canonical), secondary_dir (if it exists on disk), return_dir
    (ADR-10 deterministic return; auto-mkdir, best-effort), and each path in
    target_paths (auto-mkdir, best-effort).

    Returns:
        List of paths written. First entry is always the primary path.
    """
    slug = slug_override if slug_override is not None else _slug(result.question.text)
    filename = f"council-out-{_ts()}-{result.mode}-{slug}.md"

    lines = [
        f"# AI Council Debate: {result.question.text[:80]}",
        "",
        *_build_header(result),
        "",
        "---",
        "",
        *_build_body(result),
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
    stem_base: str | None = None,
) -> list[Path]:
    """Emit the minority/dissent report as a discrete, durable artifact (Rama 4, #15).

    Fires only when the verdict is non-unanimous (see extract_dissent). Routed to the
    same destinations as the verdict via _write_routed (canonical + secondary + return +
    targets), so a --return-dir also receives it. Returns [] when there is no dissent.

    ``stem_base`` (``<ts>-<mode>-<slug>``, from the transcript) makes this report share the
    transcript's exact <ts> — so the verdict package's minority pointer always resolves and
    a run's artifacts are one matched set. Falls back to a fresh _ts() when not supplied.
    """
    body = extract_dissent(result.synthesis)
    if body is None:
        return []

    if stem_base is not None:
        filename = f"council-minority-{stem_base}.md"
    else:
        slug = slug_override if slug_override is not None else _slug(result.question.text)
        filename = f"council-minority-{_ts()}-{result.mode}-{slug}.md"

    synth_is_label = "participant" if result.synthesizer_is_participant else "non-participant"
    lines = [
        f"# AI Council Minority Report: {result.question.text[:80]}",
        "",
        f"**Date:** {_ts('%Y-%m-%d %H:%M:%S')}",
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


# ---------------------------------------------------------------------------
# Verdict package (DRAFT-INT-1) — a caller-consumable JSON summary + human mirror.
# NOT a metrics-sidecar extension (the sidecar is telemetry; this is the deliverable).
# Consumes the seats[]/synthesis namespaces by reference; designs neither.
# ---------------------------------------------------------------------------

# Synthesis headings that carry the decision / rationale / options, by descending
# priority. Mode-dependent (pick/judge/ideas templates differ), so matched as
# case-insensitive substrings — the same D13-class heading heuristic as extract_dissent.
# Every field sourced this way is annotated source="extraction" (vs "record") so a
# transcript-free caller knows which values are parsed prose and which are structured.
# Priority order matters: the authoritative decision heading must win when a template
# carries several. "overall verdict" precedes "recommendation" so a JUDGE synthesis
# (## Overall Verdict + ## Recommendations) reports the verdict, not the first action item.
_DECISION_HEADING_MARKERS = (
    "recommended decision",  # judge synthesis-prompt primary
    "overall verdict",       # judge synthesis_output primary — must beat "recommendation"
    "recommendation",        # pick synthesis primary (## Recommendation)
    "suggested next step",   # ideas primary
    "verdict",
    "decision",
    "position",
)
_RATIONALE_HEADING_MARKERS = ("rationale", "decision criteria", "argument quality", "reasoning")
_OPTIONS_HEADING_MARKERS = (
    "alternatives considered",
    "options",
    "considered",
    "top tier",
    "idea inventory",
)


def _one_line(body: str) -> str:
    """First non-empty content line, de-bulleted and stripped of wrapping markdown emphasis."""
    for ln in body.splitlines():
        t = ln.strip().lstrip("-*#> ").strip().strip("*`_").strip()
        if t:
            return t
    return ""


def _first_by_priority(
    sections: list[tuple[str, str]], markers: tuple[str, ...]
) -> tuple[str | None, str | None]:
    """First (heading, body) whose heading contains a marker, scanned in marker priority."""
    for marker in markers:
        for heading, body in sections:
            if marker in heading.lower() and body.strip():
                return heading, body.strip()
    return None, None


def _extracted_field(
    sections: list[tuple[str, str]], markers: tuple[str, ...], *, one_line: bool = False
) -> dict:
    """A heuristic-extracted {value, source, heading} triple (source always 'extraction')."""
    heading, body = _first_by_priority(sections, markers)
    if body is None:
        return {"value": None, "source": "extraction", "heading": None}
    return {"value": _one_line(body) if one_line else body, "source": "extraction", "heading": heading}


def _extracted_options(sections: list[tuple[str, str]]) -> dict:
    """Options/alternatives as a list of bullet items (source='extraction')."""
    heading, body = _first_by_priority(sections, _OPTIONS_HEADING_MARKERS)
    items: list[str] = []
    if body:
        for ln in body.splitlines():
            t = ln.strip()
            first_token = t.split(" ", 1)[0].rstrip(".")
            if t[:1] in ("-", "*") or first_token.isdigit():
                cleaned = t.lstrip("-*0123456789. ").strip()
                if cleaned:
                    items.append(cleaned)
    return {"items": items, "source": "extraction", "heading": heading}


def _verdict_summary_lines(result: DebateResult) -> list[str]:
    """Human-readable mirror block for the top of council-out (DRAFT-INT-1).

    Prose mirror of the machine-authoritative council-verdict-*.json sibling: the decision,
    dissent status, panel seated-vs-requested, verdict author, and any degradation — so a
    Lane B/operator read is self-contained without opening the JSON.
    """
    sections = _split_sections(result.synthesis)
    decision = _extracted_field(sections, _DECISION_HEADING_MARKERS, one_line=True)["value"]
    seated = sorted({r.provider for r in result.rounds[0].responses})
    requested = sorted(set(result.provider_statuses) | set(seated))
    dropped = sorted(k for k, v in result.provider_statuses.items() if v == "failed")
    dissent = "unanimous" if extract_dissent(result.synthesis) is None else "non-unanimous (see minority report)"

    lines = [
        "## Verdict Summary",
        "",
        f"**Decision:** {decision or '(not extracted — see synthesis below)'}",
        f"**Dissent:** {dissent}",
        f"**Panel seated:** {len(seated)}/{len(requested)}"
        + (f" (dropped: {', '.join(dropped)})" if dropped else ""),
        f"**Verdict author:** {_synth_label(result)}",
    ]
    if result.degraded:
        lines.append(
            f"**Degradation:** {result.degradation_summary or 'Some providers failed during the debate.'}"
        )
    lines += [
        "",
        "_Machine-readable fields are authoritative in the council-verdict-*.json sibling._",
    ]
    return lines


def _build_verdict_payload(
    result: DebateResult,
    run_id: str,
    base: str,
    filename: str,
    written: dict[str, list[Path]],
    verdict_dirs: list[Path],
) -> dict:
    """Assemble the DRAFT-INT-1 field set, sourcing every field by reference."""
    sections = _split_sections(result.synthesis)
    seats = result.metrics.seats if result.metrics else []

    seated = sorted({r.provider for r in result.rounds[0].responses})
    requested = sorted(set(result.provider_statuses) | set(seated))
    dropped = sorted(k for k, v in result.provider_statuses.items() if v == "failed")

    if extract_dissent(result.synthesis) is None:
        dissent = {"status": "unanimous", "minority_artifact": None, "gist": None, "source": "extraction"}
    else:
        # Gist from the dissent section BODY (skip the heading, which extract_dissent re-emits).
        _, dissent_body = _first_by_priority(sections, _DISSENT_HEADING_MARKERS)
        # Point at the ACTUAL emitted minority filename when the orchestrator supplied it
        # (authoritative — save_minority_report mints its own <ts> and could differ by a
        # rollover second); fall back to reconstruction only when it wasn't passed.
        minority_written = written.get("minority")
        minority_name = (
            minority_written[0].name if minority_written else f"council-minority-{base}.md"
        )
        dissent = {
            "status": "non-unanimous",
            "minority_artifact": minority_name,
            "gist": _one_line(dissent_body or "")[:280] or None,
            "source": "extraction",
        }

    verdict_author: dict = {
        "actual": result.synthesizer,
        "is_participant": result.synthesizer_is_participant,
        "source": "record",
    }
    if result.synthesis_metrics is not None:
        verdict_author["model"] = result.synthesis_metrics.synthesizer_model
        verdict_author["error_class"] = result.synthesis_metrics.error_class

    artifacts: list[dict] = [
        {"kind": kind, "filename": paths[0].name, "paths": [str(p) for p in paths]}
        for kind, paths in written.items()
        if paths
    ]
    artifacts.append(
        {
            "kind": "verdict",
            "filename": filename,
            "paths": [str(d / filename) for d in verdict_dirs],
        }
    )

    return {
        "run_id": run_id,
        "timestamp": _iso_now(),
        # No Contract-Version is stamped until the D2 deviations empty (Q7/DRAFT-INT-2, P6);
        # emit null rather than inventing one — a caller records "unversioned" honestly.
        "contract_version": None,
        "question": result.question.text,
        "mode": result.mode,
        # A completed debate returns exit 0 even on a shrunk panel (ADR-08/§1 finding); the
        # panel/degradation fields below carry the shrunk-panel truth (two-signal rule).
        "exit_semantics": 0,
        "decision": _extracted_field(sections, _DECISION_HEADING_MARKERS, one_line=True),
        "rationale": _extracted_field(sections, _RATIONALE_HEADING_MARKERS),
        "options_considered": _extracted_options(sections),
        "dissent": dissent,
        "panel": {
            "requested": requested,
            "seated": seated,
            "dropped": dropped,
            "source": "record",
            # The canonical L-CLI seats[] serialization, consumed by reference (identical to
            # the _metrics.json sidecar shape via _seat_payload — no parallel schema).
            "seats": [_seat_payload(s) for s in seats],
        },
        "verdict_author": verdict_author,
        "degradation": {
            "degraded": result.degraded,
            "summary": result.degradation_summary,  # persisted alarm text — closes G3
            # A flattened cross-seat VIEW of panel.seats[].fallback_events (closes G4) — a
            # convenience projection of the canonical data above, not a second source.
            "fallback_events": [
                {"seat": s.seat, **_fallback_payload(fe)}
                for s in seats
                for fe in s.fallback_events
            ],
            "failed_providers": dropped,
            "source": "record",
        },
        "artifacts": artifacts,
    }


def save_verdict_package(
    result: DebateResult,
    output_dir: Path,
    transcript_path: Path,
    written: dict[str, list[Path]] | None = None,
    secondary_dir: Path | None = None,
    target_paths: list[Path] | None = None,
    return_dir: Path | None = None,
) -> list[Path]:
    """Emit the DRAFT-INT-1 verdict package as a sibling of save_to_file (#26).

    A transcript-free, caller-consumable JSON summary: council-verdict-<ts>-<mode>-<slug>.json,
    routed to every destination via _write_routed (canonical + secondary + return + targets).
    The deterministic <ts> is inherited from ``transcript_path`` (single source, so the verdict
    and its transcript always share the same stem); ``written`` is the manifest of artifacts
    already emitted this run (transcript/minority/metrics), recorded in the package's artifacts[].

    Returns the paths written, canonical first.
    """
    run_id = transcript_path.stem  # council-out-<ts>-<mode>-<slug>
    base = run_id[len("council-out-"):]  # <ts>-<mode>-<slug>
    filename = f"council-verdict-{base}.json"
    verdict_dirs = _route_dirs(output_dir, secondary_dir, target_paths, return_dir)
    payload = _build_verdict_payload(result, run_id, base, filename, written or {}, verdict_dirs)
    saved = _write_routed(
        json.dumps(payload, indent=2), filename, output_dir, secondary_dir, target_paths, return_dir
    )
    # R4: the verdict is the caller's *required* deliverable. If an explicit --return-dir was
    # requested but the package did not land there, fail loud rather than yield a bare exit 0.
    # (Optional --target-project mirrors and the legacy secondary_dir stay best-effort.)
    if return_dir is not None and not any(p.parent == return_dir for p in saved):
        raise OutputRoutingError(
            f"verdict package failed to reach required return-dir: {return_dir}"
        )
    logger.info("Verdict package saved to: %s", saved[0])
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
                "backend": c.backend,
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
        payload["seats"] = [_seat_payload(seat) for seat in m.seats]
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
