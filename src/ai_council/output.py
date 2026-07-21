"""Rich console output and markdown file save for debate results."""

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.tree import Tree

from ai_council.models import (
    SEAT_STATUS_LOST,
    SEAT_STATUS_OK,
    SEAT_STATUSES,
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


@dataclass(frozen=True)
class RoutingFailure:
    """A required destination that did not receive its artifact.

    Recorded rather than raised on the spot so the caller can finish emitting every
    remaining CANONICAL artifact before failing the run — see ``OutputRoutingError``.
    """

    artifact: str
    destination: Path
    cause: str
    # The originating exception, kept so the aggregate raise can chain a real traceback.
    # Without it the accumulator path — the one the orchestrators actually use — would
    # produce an OutputRoutingError with __cause__ None while the direct path chained
    # properly, losing the root cause exactly where it is hardest to reproduce.
    error: BaseException | None = None

    def __str__(self) -> str:
        return f"{self.artifact} -> {self.destination} ({self.cause})"


class OutputRoutingError(RuntimeError):
    """A required output destination (e.g. an explicit --return-dir) was not written.

    Raised for the caller's *required* deliverables so a routing failure surfaces loudly
    instead of yielding a bare exit 0 (DRAFT-INT-1 R4). Optional mirrors stay best-effort.

    Carries EVERY failure in the run, not just the first: the debate writers all target the
    same --return-dir, so a return-dir fault is normally common-mode, and naming one
    artifact would understate what the caller did not receive.
    """

    def __init__(self, failures: list["RoutingFailure"]) -> None:
        # str/bytes are iterable, so a bare list() silently shreds a message string into
        # one "deliverable" per character — a wrong input producing plausible-looking output
        # instead of a loud failure, which is the exact defect class this class exists to
        # remove. They are rejected before the list/tuple check because they would pass any
        # generic Iterable test. Caught live: three CLI fixtures constructed this with a
        # string and two of them still passed, asserting on text that survived the mangling.
        if isinstance(failures, (str, bytes)):
            raise TypeError(
                f"OutputRoutingError takes a list of RoutingFailure, not {type(failures).__name__}"
            )
        if not isinstance(failures, (list, tuple)):
            raise TypeError(
                f"OutputRoutingError takes a list of RoutingFailure, not {type(failures).__name__}"
            )
        bad = [f for f in failures if not isinstance(f, RoutingFailure)]
        if bad:
            raise TypeError(
                "OutputRoutingError takes a list of RoutingFailure, got element of type "
                f"{type(bad[0]).__name__}"
            )
        self.failures = list(failures)
        joined = "; ".join(str(f) for f in self.failures)
        noun = "deliverable" if len(self.failures) == 1 else "deliverables"
        super().__init__(
            f"required return-dir unusable — {len(self.failures)} {noun} not delivered: {joined}"
        )


def raise_for_routing_failures(failures: list[RoutingFailure]) -> None:
    """Fail the run if any required destination was missed; no-op when the list is empty.

    The single aggregate raise point. Callers accumulate across every writer, emit every
    canonical artifact, then call this ONCE at the end. Never call it from a ``finally`` —
    an exception raised there would mask the original.
    """
    if failures:
        # Chain the first underlying error so the traceback still reaches the real cause
        # (PermissionError, NotADirectoryError, ...). Every cause is preserved as text in
        # the message and on RoutingFailure.error; only the chained one can be a single
        # exception, and the first failure is the one that started the run down this path.
        raise OutputRoutingError(failures) from next(
            (f.error for f in failures if f.error is not None), None
        )


def _write_routed(
    content: str,
    filename: str,
    output_dir: Path,
    secondary_dir: Path | None,
    target_paths: list[Path] | None,
    return_dir: Path | None,
    *,
    artifact: str,
    return_dir_required: bool,
    routing_failures: list[RoutingFailure] | None = None,
) -> list[Path]:
    """Write `content` as `filename` to the canonical dir plus any optional routes.

    The canonical `output_dir` is ALWAYS written first (ADR-10 / ADR-43: the return
    is a copy/route, never a replacement). Then, in order:
      - `secondary_dir`: legacy mirror, written only if it already exists on disk;
      - `return_dir`: ADR-10 deterministic return (auto-mkdir), verified after the write;
      - each `target_paths` dir: ADR-43 per-invocation mirror (auto-mkdir, best-effort).

    `return_dir_required` states whether the deterministic return is a REQUIRED deliverable
    or a best-effort copy. It is an explicit decision at every call site, never inferred.
    When required, a miss is reported one of two ways (the caller chooses):

      - `routing_failures` supplied -> the miss is appended and this call returns normally,
        so the caller can finish emitting its remaining canonical artifacts and raise once
        with the aggregate via ``raise_for_routing_failures``. This is what orchestrators do.
      - `routing_failures` omitted -> ``OutputRoutingError`` is raised before returning.
        The fallback for a direct caller that opts out of collecting: a required miss must
        never be silent, whichever mode is in play.

    `artifact` names the deliverable in that report. Best-effort destinations
    (`secondary_dir`, `target_paths`) are never reported this way — they only warn.

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

    pending: RoutingFailure | None = None
    pending_exc: BaseException | None = None
    if return_dir is not None:
        cause: str | None = None
        try:
            return_dir.mkdir(parents=True, exist_ok=True)
            return_path = return_dir / filename
            return_path.write_text(content, encoding="utf-8")
            # Verify it landed: a raised write is not the only way to miss a destination.
            # is_file(), not exists() — exists() is also true for a directory. Deliberately
            # no size assertion: write_text uses text mode, so \n -> \r\n on Windows and
            # st_size never equals len(content.encode()) for a multi-line artifact.
            if return_path.is_file():
                logger.info("Deterministic return written to: %s", return_path)
                saved.append(return_path)
            else:
                cause = "write reported success but no file at the destination"
        except Exception as exc:
            cause = f"{type(exc).__name__}: {exc}"
            pending_exc = exc

        if cause is not None:
            if return_dir_required:
                logger.error("Required return-dir write failed for %s: %s", return_dir, cause)
                pending = RoutingFailure(artifact, return_dir, cause, pending_exc)
            else:
                logger.warning("Return-dir write failed for %s: %s", return_dir, cause)

    for target_dir in target_paths or []:
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / filename
            target_path.write_text(content, encoding="utf-8")
            logger.info("Mirrored to: %s", target_path)
            saved.append(target_path)
        except Exception as exc:
            logger.warning("Mirror write failed for %s: %s", target_dir, exc)

    # Deferred to here so a required-return miss still lets the best-effort mirrors run.
    if pending is not None:
        if routing_failures is not None:
            routing_failures.append(pending)
        else:
            raise OutputRoutingError([pending]) from pending_exc

    return saved


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
        failed = _dropped_providers(result.provider_statuses)
        degradation_note = result.degradation_summary or "Some providers failed during the debate."
        lines.append(f"**Status:** DEGRADED — {degradation_note}")
        if failed:
            lines.append(f"**Failed providers:** {', '.join(failed)}")

    # Provider notes: retried-and-recovered + dropped providers
    retried = sorted({r.provider for rnd in result.rounds for r in rnd.responses if r.was_retry})
    failed_providers = _dropped_providers(result.provider_statuses)
    provider_note_parts: list[str] = []
    for p in retried:
        provider_note_parts.append(f"{p} retried (timeout, recovered)")
    for p in failed_providers:
        # A lost seat was not "skipped" — it answered, then went missing (P1-8).
        if result.provider_statuses.get(p) == SEAT_STATUS_LOST:
            provider_note_parts.append(f"{p} dropped mid-debate")
        else:
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
    routing_failures: list[RoutingFailure] | None = None,
) -> list[Path]:
    """Save the full debate transcript as a markdown file.

    Pure orchestration (audit A4): filename derivation, content assembly via
    ``_build_header``/``_build_body``, routed write, metrics trigger. Writes to
    output_dir (always, canonical), secondary_dir (if it exists on disk), return_dir
    (ADR-10 deterministic return; auto-mkdir, REQUIRED per R4), and each path in
    target_paths (auto-mkdir, best-effort).

    An explicit ``return_dir`` is a required deliverable (#35): a write that does not
    land there is reported into ``routing_failures`` when the caller is accumulating, and
    raises ``OutputRoutingError`` when it is not. Never silently dropped either way.

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

    # Direct mode (no caller accumulator) used to let _write_routed raise HERE, before the
    # canonical metrics sidecar was written -- so a required return-dir failure cost a
    # canonical artifact, violating this lane's own invariant that every canonical write
    # lands before any raise. Accumulate locally either way, emit the sidecar, then raise
    # at the end of the function. Accumulating callers are unaffected: their list is used
    # directly and they still own the aggregate raise.
    accumulating = routing_failures is not None
    # Narrowed on routing_failures directly, not on `accumulating` -- mypy does not carry
    # a narrowing through an intermediate bool.
    sink: list[RoutingFailure] = routing_failures if routing_failures is not None else []

    saved = _write_routed(
        "\n".join(lines),
        filename,
        output_dir,
        secondary_dir,
        target_paths,
        return_dir,
        artifact="transcript",
        return_dir_required=True,
        routing_failures=sink,
    )
    logger.info("Debate saved to: %s", saved[0])

    if result.metrics:
        try:
            _save_metrics_json(result, saved[0])
        except Exception:
            # #63: telemetry loss must not suppress the caller's deliverables. Before this,
            # a sidecar failure propagated out of save_to_file and the orchestrator never
            # reached save_verdict_package, so no contract_version package was emitted at all.
            # logger.exception, not .error: this also covers the internal assert and any
            # malformed-payload TypeError — real bugs whose traceback must stay visible.
            logger.exception("Metrics sidecar write failed alongside %s", saved[0])
            # Report it MACHINE-READABLY, not in the log alone — a consumer of the verdict
            # package would otherwise see a complete package with no metrics entry and no way
            # to tell "metrics failed" from "metrics never produced", which is the same
            # silent-failure shape this contract exists to remove. Carried on the existing
            # #26 exit-0-plus-degradation two-signal: _build_verdict_payload already
            # serializes these two fields into the package's degradation block, so this
            # needs no new field, flag, or CONTRACT entry, and exit_semantics stays 0.
            note = f"metrics sidecar not written for {saved[0].name}"
            result.degraded = True
            result.degradation_summary = (
                f"{result.degradation_summary}; {note}" if result.degradation_summary else note
            )

    # Every canonical artifact is now on disk. Only a direct-mode caller raises here; an
    # accumulating caller owns the single aggregate raise after ALL its writers have run.
    if not accumulating:
        raise_for_routing_failures(sink)

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
    routing_failures: list[RoutingFailure] | None = None,
) -> list[Path]:
    """Emit the minority/dissent report as a discrete, durable artifact (Rama 4, #15).

    Fires only when the verdict is non-unanimous (see extract_dissent). Routed to the
    same destinations as the verdict via _write_routed (canonical + secondary + return +
    targets). Returns [] when there is no dissent.

    An explicit ``return_dir`` is a required deliverable and carries the same R4 guarantee
    as the verdict (#35): a write that does not land there is reported into
    ``routing_failures`` when the caller is accumulating, and raises ``OutputRoutingError``
    when it is not. Until #35 this docstring claimed "so a --return-dir also receives it",
    which the code did not honour — the write was best-effort and a miss was swallowed.

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
        "\n".join(lines),
        filename,
        output_dir,
        secondary_dir,
        target_paths,
        return_dir,
        artifact="minority report",
        return_dir_required=True,
        routing_failures=routing_failures,
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
    # NO bare "considered". It matched as a SUBSTRING, so "## Risks Considered" qualified
    # as an options heading. Harmless while the scan stopped at the first match; once #60
    # made it CONTINUE past a prose-only section, a later "...Considered" section could be
    # promoted over the question's real options -- risks served to a caller as the
    # decision's options_considered. Under-match deliberately: a heading like "Approaches
    # Considered" now falls through to the question fallback, which is honestly empty or
    # honestly the question's own options. Over-matching produced plausible wrongness that
    # a consumer cannot detect, and this field is on the delegation surface.
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


# #77: the list marker is matched as a GRAMMAR, not a character set. The old
# `lstrip("-*0123456789. ")` was a character-set strip, so it kept eating past the marker
# into the option's own text — `- 3D printing` became `D printing` and `- 2026 roadmap`
# became `roadmap`. A regex consumes the exact marker and nothing else. Whitespace after
# the marker is REQUIRED, which is also what keeps a bare `2026 roadmap` line (no marker)
# from being read as a numbered item and shorn of its year.
# `[0-9]{1,9}`, not `\d+`: an ordered-list marker is 1-9 ASCII digits (CommonMark §5.2).
# `\d+` accepted `1234567890. ordinary prose` as a list item — fabricating an option out of
# prose that merely opens with a long number — and `\d` also matches non-ASCII digits.
# CommonMark line endings only: LF, CR, CRLF. `str.splitlines()` also breaks on U+2028,
# U+2029 and U+0085, none of which end a Markdown line.
_LINE_BREAK_RE = re.compile(r"\r\n|\r|\n")

# `[ \t]+`, not `\s+`: a marker separator is an ASCII space or tab. `\s` also matches
# Unicode spaces, so a hyphen followed by U+00A0 parsed as a list item and fabricated
# an option out of ordinary prose (terra pass 4).
_BULLET_RE = re.compile(r"^(?:[-*+]|[0-9]{1,9}[.)])[ \t]+(?P<text>.*)$")

# A thematic break is not a list item. `---` fails _BULLET_RE anyway (no space after the
# marker), but the spaced form `* * *` parses as a `*` bullet whose payload is `* *`. The
# old character-set strip dropped that incidentally; matching the marker exactly makes the
# guard explicit — honest-empty beats a junk option on the delegation surface.
# `[ \t]*`, not `\s*`, for the same reason as _BULLET_RE: NBSP-separated asterisks are
# option payload, not a separator, and reading them as a break DELETED the whole option
# (terra pass 5). Narrowing one regex and not the other was an inconsistency of mine.
_THEMATIC_BREAK_RE = re.compile(r"^([-*_])(?:[ \t]*\1){2,}$")

# ASCII punctuation is what a markdown backslash escape may precede (CommonMark §2.4).
_ESCAPABLE = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

# Emphasis nesting deeper than this is not real prose. The cap keeps the closer/opener
# search bounded, so unwrapping stays linear in the length of the option: a delimiter-heavy
# line (a model may emit up to 16k tokens on one line) must not stall verdict generation.
_MAX_EMPHASIS_NESTING = 64


def _is_punctuation(ch: str) -> bool:
    """CommonMark punctuation — ASCII punctuation plus Unicode P* and S* categories.

    The S* (symbol) categories are load-bearing, not pedantry: without them a currency or
    math symbol is treated as an ordinary letter, so ``a*€*b`` looked like real emphasis
    and lost both asterisks (terra pass 4).
    """
    return bool(ch) and (ch in _ESCAPABLE or unicodedata.category(ch)[0] in "PS")


def _pair_code_spans(text: str) -> dict[int, tuple[int, int, int]]:
    """Map each code-span opener offset to ``(content_start, content_end, close_end)``.

    Backtick runs are collected MAXIMALLY and paired only with a run of exactly equal
    length (CommonMark §6.1). Collecting maximal runs up front is what makes a longer run
    unusable as a shorter fence's closer — searching for the fence substring instead let a
    one-backtick opener pair with the third backtick of a ```` ``` ```` run and swallow the
    other two (terra review pass 2).

    Two linear passes, no suffix rescanning: an earlier version called ``find()`` per
    unmatched run, which re-walked the tail once per run.
    """
    # Runs carry whether their first backtick was escaped. An escaped run cannot OPEN a
    # span — `\`` is a literal backtick, and treating it as an opener let it swallow the
    # real `` `__init__` `` after it (terra pass 3). It CAN still close one: backslash
    # escapes do not apply inside a code span, so `` `C:\` `` is the path `C:\`, and
    # skipping that closing backtick corrupted the payload (terra pass 4).
    runs: list[tuple[int, int, bool]] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "\\" and i + 1 < n and text[i + 1] in _ESCAPABLE:
            if text[i + 1] != "`":
                i += 2  # some other escaped punctuation — irrelevant to code spans
                continue
            i += 1  # step onto the backtick; the run below records it, flagged as escaped
        if text[i] == "`":
            run_end = i
            while run_end < n and text[run_end] == "`":
                run_end += 1
            runs.append((i, run_end, text[i - 1 : i] == "\\"))
            i = run_end
        else:
            i += 1

    # Index runs by their FULL length: that is the length a run has when it CLOSES a span,
    # where backslash escapes do not apply. Splitting a backslash-adjacent run to model the
    # escape mis-sized the closing fence of `` ``C:\`` `` and dropped the backslash from the
    # payload (terra pass 6). The escape only ever affects the OPENING side, handled below.
    by_length: dict[int, list[int]] = {}
    for idx, (start, end, _escaped) in enumerate(runs):
        by_length.setdefault(end - start, []).append(idx)
    cursors = dict.fromkeys(by_length, 0)

    spans: dict[int, tuple[int, int, int]] = {}
    idx = 0
    while idx < len(runs):
        start, end, was_escaped = runs[idx]
        # Only the FIRST backtick of the run is escaped, so it is literal and cannot open;
        # any remainder still can. `\``x``` opens a two-backtick span after one literal
        # backtick, while `\`` alone opens nothing (terra passes 3 and 5).
        open_start = start + 1 if was_escaped else start
        if open_start >= end:
            idx += 1  # nothing left to open with — it may still serve as a closer
            continue
        siblings = by_length.get(end - open_start, [])
        cursor = cursors.get(end - open_start, 0)
        while cursor < len(siblings) and siblings[cursor] <= idx:
            cursor += 1
        cursors[end - open_start] = cursor
        if cursor >= len(siblings):
            idx += 1  # no equal-length closer anywhere after it — this run is literal
            continue
        # An opener takes the FIRST equal-length run after it (CommonMark §6.1), so the
        # OUTER pair wins: in `x ``y`` z` the outer single backticks form the span and the
        # inner double run is content. Pairing whichever closed first inverted that and
        # ate the outer backticks' payload.
        closer = siblings[cursor]
        spans[open_start] = (end, runs[closer][0], runs[closer][1])
        idx = closer + 1
        continue
    return spans


def _unwrap_emphasis(text: str) -> str:
    """Remove PAIRED markdown emphasis delimiters anywhere in ``text``.

    ``_one_line``'s ``.strip("*`_")`` idiom is edge-only, so ``**Alpha** — fast`` kept its
    interior ``**`` (``Alpha** — fast``). This unwraps real delimiter pairs wherever they
    sit, including nested ones (``**_x_**``).

    A single left-to-right scan rather than a regex, for three reasons a regex got wrong
    (terra review, #77):

    * **Escapes** — ``\\*literal\\*`` is an author asking for literal asterisks, not
      emphasis. An escaped delimiter can never open or close a span; it de-escapes to the
      bare character, which is what the author meant to be read.
    * **Code spans** — a backtick span is ATOMIC. ```` `__init__` ```` must survive whole;
      a regex that unwrapped the backticks then re-scanned the result ate the dunder and
      produced ``init``. Payload loss on the delegation surface, exactly what #77 forbids.
    * **Linear time** — a lazy ``(.+?)`` between delimiters retries from every candidate
      opener when no closer exists, which is quadratic; ~30k characters of ``" *a"`` took
      seconds. Scanning once with a depth-capped stack cannot backtrack.

    Only EQUAL-LENGTH delimiter runs pair (``**x**`` yes, ``**x*`` no). Anything unpaired is
    emitted verbatim — an honest ``**`` in the output beats a payload character silently
    removed, and this field is read by a consuming repo as the council's own wording.
    """
    code_spans = _pair_code_spans(text)
    parts: list[dict] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            if i + 1 < n and text[i + 1] in _ESCAPABLE:
                parts.append({"emit": text[i + 1]})  # de-escaped, and inert as a delimiter
                i += 2
            else:
                # A backslash that escapes nothing is ordinary payload — `C:\Users`. It must
                # still consume a character: the fallback branch below stops AT a backslash
                # without advancing, so reaching it here spun forever and hung verdict
                # generation on any Windows path (terra pass 4). Every branch must progress.
                parts.append({"emit": "\\"})
                i += 1
        elif ch == "`":
            span = code_spans.get(i)
            if span is None:
                run_end = i
                while run_end < n and text[run_end] == "`":
                    run_end += 1
                parts.append({"emit": text[i:run_end]})  # unpaired run — keep it, do not guess
                i = run_end
            else:
                content_start, content_end, close_end = span
                parts.append({"emit": text[content_start:content_end]})  # atomic span content
                i = close_end
        elif ch in "*_":
            run_end = i
            while run_end < n and text[run_end] == ch:
                run_end += 1
            run = text[i:run_end]
            before = text[i - 1] if i > 0 else ""
            after = text[run_end] if run_end < n else ""
            # CommonMark §6.2 flanking. A whitespace-only test deleted literal payload:
            # `a*.*b` has no emphasis (the run is followed by punctuation but preceded by a
            # letter, so it cannot open) yet collapsed to `a.b` (terra pass 3).
            before_ws, after_ws = (not before) or before.isspace(), (not after) or after.isspace()
            before_punct, after_punct = _is_punctuation(before), _is_punctuation(after)
            left_flanking = not after_ws and (not after_punct or before_ws or before_punct)
            right_flanking = not before_ws and (not before_punct or after_ws or after_punct)
            can_open, can_close = left_flanking, right_flanking
            if ch == "_":
                # Intra-word `_` is payload, not emphasis: `snake_case_name` must survive.
                can_open = left_flanking and (not right_flanking or before_punct)
                can_close = right_flanking and (not left_flanking or after_punct)
            parts.append({"emit": run, "char": ch, "len": len(run), "open": can_open, "close": can_close})
            i = run_end
        else:
            run_end = i
            while run_end < n and text[run_end] not in "\\`*_":
                run_end += 1
            parts.append({"emit": text[i:run_end]})
            i = run_end

    openers: list[dict] = []

    def push(part: dict) -> None:
        """Every opener goes through here so the depth cap can never be bypassed.

        A delimiter that can both close and open reached ``append`` directly before, so a
        stream of never-matching flanked runs (``!_!*!_!*…``) grew the stack without bound
        and each reverse search walked all of it — quadratic on linear input (terra pass 2).
        """
        openers.append(part)
        del openers[:-_MAX_EMPHASIS_NESTING]

    for part in parts:
        if "char" not in part:
            continue
        if part["close"]:
            for k in range(len(openers) - 1, -1, -1):
                candidate = openers[k]
                if candidate["char"] == part["char"] and candidate["len"] == part["len"]:
                    candidate["emit"] = part["emit"] = ""  # a real pair — drop both runs
                    del openers[k:]  # openers left dangling inside the span stay literal
                    break
            else:
                if part["open"]:
                    push(part)
            continue
        if part["open"]:
            push(part)
    return "".join(part["emit"] for part in parts).strip()


def _top_level_bullets(body: str | None) -> list[str]:
    """Top-level bullet/numbered items in a section body.

    Accepts every markdown list grammar — ``-``, ``*``, ``+``, ``1.``, ``1)`` — and removes
    the exact marker only, never a payload character (#77). A nested (indented) sub-bullet —
    e.g. an ideas entry's ``Who endorsed it`` annotation — is dropped rather than scooped as
    its own option. Markdown emphasis is unwrapped as paired delimiters so no ``**`` leaks
    into the field.

    This feeds ``options_considered`` on the delegation surface, so a mangled item is read
    by a consuming repo as the council's own wording — hence exactness over tolerance.
    """
    items: list[str] = []
    # `_LINE_BREAK_RE`, not `splitlines()`: the latter also breaks on U+2028/U+2029/U+0085,
    # which are not CommonMark line endings, so inline payload containing one was split into
    # a fresh "line" whose tail then parsed as a list item (terra pass 5).
    for raw in _LINE_BREAK_RE.split(body or ""):
        if raw[:1] in (" ", "\t"):  # nested sub-bullet — not a top-level option
            continue
        # Only the TAIL is trimmed. `raw.strip()` also removed leading Unicode whitespace,
        # so `- prose` had its NBSP stripped and the exposed hyphen fabricated an
        # option (terra pass 5). Leading ASCII space/tab is already handled just above, so
        # a top-level marker must sit at offset 0.
        stripped = raw.rstrip()
        if _THEMATIC_BREAK_RE.match(stripped):
            continue
        match = _BULLET_RE.match(stripped)
        if match is None:
            continue
        payload = match.group("text").strip(" \t")
        # Re-test AFTER marker removal: `- * * *` is a list-wrapped thematic break, so the
        # line-level guard above does not see it (terra review, #77). The old character-set
        # strip discarded it incidentally; missing this reintroduced a fabricated option.
        if _THEMATIC_BREAK_RE.match(payload):
            continue
        cleaned = _unwrap_emphasis(payload)
        if cleaned:
            items.append(cleaned)
    return items


def _options_with_items(sections: list[tuple[str, str]]) -> tuple[str | None, list[str]]:
    """First options-ish section that actually yields top-level bullets.

    Scans in marker priority like ``_first_by_priority``, but keeps looking when a matching
    section holds only prose (#60) — a *later* section may carry the real list, e.g. a
    synthesis with a prose ``## Alternatives Considered`` followed by a bulleted
    ``## Options``. Taking the first matching heading unconditionally would skip past the
    live options and fall through to the question's staler ones.

    Returns the first matching heading even when nothing yields items, so the caller can
    still report which heading was seen rather than claiming none existed.
    """
    first_heading: str | None = None
    for marker in _OPTIONS_HEADING_MARKERS:
        for heading, body in sections:
            if marker in heading.lower() and body.strip():
                if first_heading is None:
                    first_heading = heading
                items = _top_level_bullets(body.strip())
                if items:
                    return heading, items
    return first_heading, []


def _extracted_field(
    sections: list[tuple[str, str]], markers: tuple[str, ...], *, one_line: bool = False
) -> dict:
    """A heuristic-extracted {value, source, heading} triple (source always 'extraction')."""
    heading, body = _first_by_priority(sections, markers)
    if body is None:
        return {"value": None, "source": "extraction", "heading": None}
    return {"value": _one_line(body) if one_line else body, "source": "extraction", "heading": heading}


def _extracted_options(
    sections: list[tuple[str, str]],
    question_sections: list[tuple[str, str]] | None = None,
) -> dict:
    """Options/alternatives as a list of top-level bullet items (source='extraction').

    Sources the options section from the synthesis first — an ideas verdict's ``## Top Tier``
    (or a synthesis that carries an explicit ``## Alternatives Considered``). When the synthesis
    yields no options — because it carries no matching heading, the pick synthesis template
    (``prompts.synthesis``) prescribing none (#40), or because the heading it does carry holds
    only prose (#60) — falls back to the debate QUESTION's own ``## Options`` section, which is
    where a pick debate's alternatives actually live. The fallback is adopted only when it
    yields options, so it can never replace a real synthesis heading with nothing.

    Only TOP-LEVEL bullets are kept; see ``_top_level_bullets``.
    """
    # #60: the gate is "produced no OPTIONS", not "has no SECTION". It used to be
    # `not body`, so an options heading whose body is prose short-circuited the fallback
    # and returned items=[] under a heading that listed nothing — reading as "alternatives
    # were considered and there were none".
    heading, items = _options_with_items(sections)
    if not items and question_sections is not None:
        fallback_heading, fallback_items = _options_with_items(question_sections)
        # Adopt the fallback ONLY if it actually yielded options. Rebinding unconditionally
        # would clobber a valid synthesis heading with None whenever the question carries
        # no ## Options of its own.
        if fallback_items:
            heading, items = fallback_heading, fallback_items
    return {"items": items, "source": "extraction", "heading": heading}


def _dropped_providers(provider_statuses: dict[str, str]) -> list[str]:
    """Seats absent from the final round — panel.dropped / degradation.failed_providers (P1-8).

    Derived as "not ok" rather than "== failed": a seat that answered an earlier round and then
    went missing ("lost") is just as absent from the verdict-bearing round as one that never
    answered, and both belong in the caller's dropped list. Before this, only never-answered
    seats were counted, so mid-debate loss reached the contract surface as an empty list.

    An unrecognized value is counted as dropped and logged: the pessimistic direction matches
    the producer's own seeding, so a future vocabulary drift degrades loudly instead of
    silently promoting an unknown seat to seated.
    """
    unknown = sorted(set(provider_statuses.values()) - SEAT_STATUSES)
    if unknown:
        logger.warning(
            "Unrecognized provider_statuses value(s) %s — counting as dropped. "
            "Extend SEAT_STATUSES in models.py rather than emitting a bare string.",
            ", ".join(unknown),
        )
    return sorted(k for k, v in provider_statuses.items() if v != SEAT_STATUS_OK)


def _synthesis_failed(result: DebateResult) -> bool:
    """True when this result stands in for a synthesis that never returned (P1-9).

    Read off the STRUCTURED observability field, not by sniffing the synthesis body for a
    marker string — the codebase already has four string-matching classifiers that misfire,
    and this one has real data available. ``error_class`` is "none" on every success path and
    a classified token only on the preservation path.
    """
    return result.synthesis_metrics is not None and result.synthesis_metrics.error_class != "none"


def _verdict_summary_lines(result: DebateResult) -> list[str]:
    """Human-readable mirror block for the top of council-out (DRAFT-INT-1).

    Prose mirror of the machine-authoritative council-verdict-*.json sibling: the decision,
    dissent status, panel seated-vs-requested, verdict author, and any degradation — so a
    Lane B/operator read is self-contained without opening the JSON.

    On the P1-9 preservation path there IS no verdict package, so this renders an explicit
    no-verdict block instead: a preserved transcript claiming "Dissent: unanimous" and
    pointing at a council-verdict-*.json sibling that was deliberately never written is the
    same defect #63 guards against — naming a file the consumer then fails to find.
    """
    if _synthesis_failed(result):
        seated = sorted({r.provider for r in result.rounds[0].responses})
        requested = sorted(set(result.provider_statuses) | set(seated))
        return [
            "## Verdict Summary",
            "",
            "**No verdict was produced.** The synthesizer did not return one, so this run "
            "emitted no verdict package and no minority report.",
            f"**Panel seated:** {len(seated)}/{len(requested)}",
            f"**Degradation:** {result.degradation_summary or 'Synthesis failed.'}",
            "",
            "_The debate rounds below are the authoritative record of this run._",
        ]
    sections = _split_sections(result.synthesis)
    decision = _extracted_field(sections, _DECISION_HEADING_MARKERS, one_line=True)["value"]
    seated = sorted({r.provider for r in result.rounds[0].responses})
    requested = sorted(set(result.provider_statuses) | set(seated))
    dropped = _dropped_providers(result.provider_statuses)
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
    filename: str,
    written: dict[str, list[Path]],
    guaranteed_dirs: list[Path],
) -> dict:
    """Assemble the DRAFT-INT-1 field set, sourcing every field by reference.

    ``guaranteed_dirs`` are the destinations known to receive the verdict when the call
    returns normally (canonical, always written; and an explicit return_dir, which R4
    verifies-or-reports in ``_write_routed``). Best-effort mirrors are deliberately excluded
    so the manifest never claims a path that a best-effort write may have skipped.
    """
    sections = _split_sections(result.synthesis)
    seats = result.metrics.seats if result.metrics else []

    seated = sorted({r.provider for r in result.rounds[0].responses})
    requested = sorted(set(result.provider_statuses) | set(seated))
    dropped = _dropped_providers(result.provider_statuses)

    if extract_dissent(result.synthesis) is None:
        dissent = {"status": "unanimous", "minority_artifact": None, "gist": None, "source": "extraction"}
    else:
        # Gist from the dissent section BODY (skip the heading, which extract_dissent re-emits).
        _, dissent_body = _first_by_priority(sections, _DISSENT_HEADING_MARKERS)
        # Point ONLY at an ACTUAL emitted minority file (authoritative — save_minority_report
        # mints its own <ts>). Never fabricate a name: if the caller emitted no minority
        # report, the pointer is null (status + gist still convey the dissent) rather than a
        # dangling path. The orchestrator always supplies it for a non-unanimous verdict.
        minority_written = written.get("minority")
        dissent = {
            "status": "non-unanimous",
            "minority_artifact": minority_written[0].name if minority_written else None,
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

    # Never advertise a path that is not on disk (#63). The verdict package is the
    # inter-repo delegation surface at Contract-Version 1.0: a manifest naming a file a
    # consumer then fails to find is worse than the original defect. The orchestrator
    # already guards the one known case (a degraded metrics sidecar); this filter makes it
    # structurally impossible regardless of who populates `written`.
    artifacts: list[dict] = []
    for kind, paths in written.items():
        present = [p for p in paths if p.exists()]
        if len(present) != len(paths):
            missing = [str(p) for p in paths if p not in present]
            logger.warning(
                "Verdict manifest: omitting %d %s path(s) not on disk: %s",
                len(missing), kind, ", ".join(missing),
            )
        if present:
            artifacts.append(
                {"kind": kind, "filename": present[0].name, "paths": [str(p) for p in present]}
            )
    artifacts.append(
        {
            "kind": "verdict",
            "filename": filename,
            "paths": [str(d / filename) for d in guaranteed_dirs],
        }
    )

    return {
        "run_id": run_id,
        "timestamp": _iso_now(),
        # Contract-Version 1.0: stamped once the D2 deviations emptied — --file frontmatter
        # (#22) + research --return-dir (#23) shipped, so CONTRACT §7 is empty (Q7/DRAFT-INT-2).
        "contract_version": "1.0",
        "question": result.question.text,
        "mode": result.mode,
        # A completed debate returns exit 0 even on a shrunk panel (ADR-08/§1 finding); the
        # panel/degradation fields below carry the shrunk-panel truth (two-signal rule).
        "exit_semantics": 0,
        "decision": _extracted_field(sections, _DECISION_HEADING_MARKERS, one_line=True),
        "rationale": _extracted_field(sections, _RATIONALE_HEADING_MARKERS),
        "options_considered": _extracted_options(
            sections, question_sections=_split_sections(result.question.text)
        ),
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
    routing_failures: list[RoutingFailure] | None = None,
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
    # Destinations guaranteed once this call returns normally: canonical (always written) and
    # an explicit return_dir (which _write_routed verifies-or-reports). Best-effort mirrors
    # are excluded.
    guaranteed_dirs = [output_dir] + ([return_dir] if return_dir is not None else [])
    payload = _build_verdict_payload(result, run_id, filename, written or {}, guaranteed_dirs)
    # R4: the verdict is the caller's *required* deliverable. If an explicit --return-dir was
    # requested but the package did not land there, fail loud rather than yield a bare exit 0.
    # (Optional --target-project mirrors and the legacy secondary_dir stay best-effort.)
    # _write_routed now owns that guarantee — it verifies the write on disk rather than
    # inferring success from list membership, and every deliverable inherits it.
    saved = _write_routed(
        json.dumps(payload, indent=2),
        filename,
        output_dir,
        secondary_dir,
        target_paths,
        return_dir,
        artifact="verdict package",
        return_dir_required=True,
        routing_failures=routing_failures,
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
