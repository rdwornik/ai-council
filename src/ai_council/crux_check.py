"""Bounded crux check between Round 1 and Round 2 (#18).

One LLM call names the central empirical crux from the ALREADY-ANONYMIZED Round-1 block,
a narrow headless retrieval checks it, and one canonical evidence artifact is returned for
injection into every Round-2 prompt.

Three outcomes, all non-fatal (see CruxStatus):
  - grounded              — a checkable claim was found and evidence retrieved
  - no_empirical_crux     — VALID SUCCESS: nothing checkable, so nothing was retrieved
  - retrieval_unavailable — the debate DEGRADES and proceeds without the artifact

The service NEVER raises into the debate: a failing extractor, a failing executor, or an
empty pool all resolve to retrieval_unavailable.

ADR-03: ``check`` accepts the anonymized block as a plain string and never sees
``list[ModelResponse]``, so it is structurally incapable of learning which panelist said
what. The evidence block it returns is likewise built from research prose only — never from
``merged_report``, which carries per-provider attribution headers.
"""

import logging
import re
from collections.abc import Awaitable, Callable
from enum import Enum

from ai_council.metrics import build_call_metrics
from ai_council.models import CruxArtifact, CruxStatus
from ai_council.providers.base import AIProvider, ProviderError
from ai_council.research.headless import run_research_headless
from ai_council.research.models import MergedResearchReport
from config.config_loader import AppConfig

logger = logging.getLogger(__name__)

# The extraction call is asked to answer under a `## Crux` heading. Matching the heading as
# a GRAMMAR (rather than a substring) keeps a stray "crux" mention in prose from winning.
_CRUX_HEADING_RE = re.compile(r"^#{1,6}\s+crux\b", re.IGNORECASE)
_ANY_HEADING_RE = re.compile(r"^#{1,6}\s")

# Suppressing retrieval is a consequential act, so the bar for "the model said there is no
# crux" is deliberately high: an EXACT sentinel, or an explicit crux-absence PHRASE.
#
# Matched exactly (after stripping trailing punctuation), never as a prefix. "none" as a
# prefix swallowed "Nonetheless, deployments fail more often" and "None of the benchmarks
# met the target" — both valid empirical claims (terra pass-2). A prefix rule on a short
# common word cannot distinguish a sentinel from the first word of a sentence.
_NO_CRUX_EXACT = ("none", "n/a", "na", "no crux")

# Matched as prefixes — each is long and unambiguous enough that no real claim opens with
# it. Deliberately NOT here: a bare "there is no". "There is no statistically significant
# difference between A and B" is textbook checkable, and generic negation is a property of
# many valid claims; only crux-absence phrasing may suppress retrieval.
_NO_CRUX_PREFIXES = (
    "no empirical crux",
    "no empirical disagreement",
    "no checkable claim",
    "no checkable empirical",
    "no factual disagreement",
    "there is no empirical crux",
    "there is no checkable",
)

# A refusal or an admission of uncertainty UNDER a well-formed heading is an extraction
# FAILURE, not a claim. Without this, "I cannot determine a crux because the input is
# incomplete" was sent to retrieval as if it were a factual claim, and a hit would have
# been reported GROUNDED (terra pass-2).
_REFUSAL_MARKERS = (
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "unable to determine",
    "cannot determine",
    "can't determine",
    "i do not have",
    "i don't have",
    "i'm sorry",
    "i am sorry",
    "as an ai",
    "insufficient information",
    "not enough information",
)

# Hard bound on the injected block. Round-2 prompts already carry the full anonymized
# Round-1 block; unbounded evidence would re-bill every panelist for a runaway retrieval.
# _MAX_EVIDENCE_CHARS bounds the research body; _MAX_ARTIFACT_CHARS bounds the ASSEMBLED
# artifact (header + claim + body + sources + footer), because bounding only the body let
# a long claim or source list push the real injected size past the cap (terra HIGH-2).
_MAX_EVIDENCE_CHARS = 4000
_MAX_ARTIFACT_CHARS = 6000
_MAX_CLAIM_CHARS = 400
_MAX_SOURCES_LISTED = 5


class ParseState(str, Enum):
    """How to read an extraction response.

    CLAIM and NO_CRUX are both well-formed answers. MALFORMED is an extraction FAILURE and
    must never be reported as a no-crux success — collapsing the two let a truncated or
    refused response silently skip retrieval while claiming the panel had nothing checkable
    to look up (terra HIGH-1).
    """

    CLAIM = "claim"
    NO_CRUX = "no_crux"
    MALFORMED = "malformed"


def _parse_crux(text: str) -> tuple[ParseState, str]:
    """Parse an extraction response into (state, claim).

    ``claim`` is meaningful only when state is CLAIM.
    """
    if not text or not text.strip():
        return ParseState.MALFORMED, ""

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if not _CRUX_HEADING_RE.match(line.strip()):
            continue
        # Take the first non-empty line under the heading, stopping at the next heading.
        for candidate in lines[idx + 1 :]:
            stripped = candidate.strip()
            if not stripped:
                continue
            if _ANY_HEADING_RE.match(stripped):
                # A `## Crux` heading with an empty body is a broken answer, not a no-crux
                # verdict — the model was asked to write NONE, and did not.
                return ParseState.MALFORMED, ""
            claim = stripped.strip("*_` ").strip()
            if not claim:
                return ParseState.MALFORMED, ""

            normalized = claim.lower().rstrip(".!,;: ")
            if normalized in _NO_CRUX_EXACT or normalized.startswith(_NO_CRUX_PREFIXES):
                return ParseState.NO_CRUX, ""
            if any(marker in normalized for marker in _REFUSAL_MARKERS):
                # A refusal beneath a valid heading: we did not find out whether a crux
                # exists, so this is a failure, not a no-crux verdict.
                return ParseState.MALFORMED, ""

            if len(claim) > _MAX_CLAIM_CHARS:
                claim = claim[:_MAX_CLAIM_CHARS].rstrip() + " […]"
            return ParseState.CLAIM, claim
        return ParseState.MALFORMED, ""
    # No `## Crux` heading at all: a refusal, a preamble-only reply, or a truncation.
    return ParseState.MALFORMED, ""


def _build_evidence_block(header: str, claim: str, report: MergedResearchReport) -> str:
    """Assemble the ONE canonical injectable block.

    Built from per-result prose, NOT ``report.merged_report``: the merged report carries
    "## perplexity"-style provider headers, and several research provider names collide with
    panel model names (gemini, grok, openai), which would read as panel attribution.
    """
    successful = [r for r in report.results if not r.error and r.content.strip()]
    body = "\n\n".join(r.content.strip() for r in successful)
    if len(body) > _MAX_EVIDENCE_CHARS:
        body = body[:_MAX_EVIDENCE_CHARS].rstrip() + " […]"

    parts = [header, "", f"Claim under check: {claim}", "", body]

    seen: set[str] = set()
    urls: list[str] = []
    for result in successful:
        for source in result.sources:
            if source.url and source.url not in seen:
                seen.add(source.url)
                urls.append(source.url)
    if urls:
        parts.extend(["", "Sources:"])
        parts.extend(f"- {u}" for u in urls[:_MAX_SOURCES_LISTED])

    parts.extend(
        [
            "",
            "This evidence is provided to all council members equally. Weigh it on its "
            "merits; it is retrieved material, not a council member's argument.",
        ]
    )
    artifact = "\n".join(parts)
    if len(artifact) > _MAX_ARTIFACT_CHARS:
        artifact = artifact[:_MAX_ARTIFACT_CHARS].rstrip() + " […]"
    return artifact


class CruxCheckService:
    """Bounded crux check. Constructed by the orchestrator, injected into run_debate."""

    def __init__(
        self,
        extractor: AIProvider,
        config: AppConfig,
        *,
        executor: Callable[..., Awaitable[MergedResearchReport | None]] = run_research_headless,
    ) -> None:
        self._extractor = extractor
        self._config = config
        self._executor = executor

    async def check(self, question_text: str, anon_block: str) -> CruxArtifact:
        """Identify the crux, check it, and return the artifact. Never raises."""
        cfg = self._config.crux_check
        if cfg is None:  # defensive: the builder refuses to construct without config
            return CruxArtifact(
                status=CruxStatus.RETRIEVAL_UNAVAILABLE, detail="crux_check not configured"
            )

        prompt = cfg.extraction_prompt.format(question=question_text, anon_block=anon_block)

        try:
            response = await self._extractor.generate(prompt, round_number=-1)
        except ProviderError as exc:
            logger.warning("Crux extraction call failed: %s", exc)
            return CruxArtifact(
                status=CruxStatus.RETRIEVAL_UNAVAILABLE, detail=f"extraction failed: {exc}"
            )

        call_metrics = build_call_metrics(response, self._config.models, round_number=-1)

        state, claim = _parse_crux(response.content or "")

        if state is ParseState.MALFORMED:
            # An unreadable extraction is a FAILURE, not a no-crux determination. Reporting
            # it as the latter would claim the panel had nothing checkable when in fact we
            # never found out (terra HIGH-1).
            logger.warning("Crux check: extraction response was not parseable")
            return CruxArtifact(
                status=CruxStatus.RETRIEVAL_UNAVAILABLE,
                detail="extraction response was not parseable",
                call_metrics=call_metrics,
            )

        if state is ParseState.NO_CRUX:
            logger.info("Crux check: no empirical crux in Round 1 — retrieval skipped")
            return CruxArtifact(
                status=CruxStatus.NO_EMPIRICAL_CRUX,
                detail="no checkable empirical claim in Round 1",
                call_metrics=call_metrics,
            )

        attempted = len(cfg.providers)
        try:
            report = await self._executor(
                claim,
                self._config,
                provider_names=list(cfg.providers),
                budget_sec=cfg.budget_sec,
            )
        except Exception as exc:  # noqa: BLE001 - retrieval must never abort the debate
            logger.warning("Crux retrieval raised: %s", exc)
            return CruxArtifact(
                status=CruxStatus.RETRIEVAL_UNAVAILABLE,
                crux_claim=claim,
                detail=f"retrieval error: {exc}",
                providers_attempted=attempted,
                call_metrics=call_metrics,
            )

        if report is None:
            return CruxArtifact(
                status=CruxStatus.RETRIEVAL_UNAVAILABLE,
                crux_claim=claim,
                detail="no research provider available or budget exceeded",
                providers_attempted=attempted,
                call_metrics=call_metrics,
            )

        succeeded = sum(1 for r in report.results if not r.error and r.content.strip())
        if succeeded == 0:
            return CruxArtifact(
                status=CruxStatus.RETRIEVAL_UNAVAILABLE,
                crux_claim=claim,
                detail="all research providers failed",
                providers_attempted=attempted or len(report.results),
                call_metrics=call_metrics,
            )

        logger.info("Crux check grounded: %s (%d provider(s))", claim, succeeded)
        return CruxArtifact(
            status=CruxStatus.GROUNDED,
            crux_claim=claim,
            evidence_block=_build_evidence_block(cfg.injection_header, claim, report),
            sources_count=report.total_sources,
            providers_succeeded=succeeded,
            providers_attempted=attempted or len(report.results),
            call_metrics=call_metrics,
        )


def build_crux_check_service(
    config: AppConfig, synthesizer: AIProvider
) -> CruxCheckService | None:
    """Build the service from config, or None when the step is not configured.

    Deliberately mirrors ``build_seat_router``: the orchestrator calls this unconditionally
    and threads the (possibly None) result into run_debate, so an unconfigured repo runs
    exactly as it did before.

    The extractor is the already-selected synthesizer — a non-participant by default, so no
    panelist gains an asymmetric role heading into Round 2.
    """
    cfg = config.crux_check
    if cfg is None:
        return None
    if not cfg.providers:
        logger.info("Crux check configured with no providers — step disabled")
        return None
    if not cfg.extraction_prompt.strip():
        logger.warning("Crux check has no extraction_prompt — step disabled")
        return None
    return CruxCheckService(synthesizer, config)
