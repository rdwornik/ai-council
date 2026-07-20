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

# The model was told to answer exactly "NONE", but models paraphrase. These are checked as
# prefixes of the claim line, mirroring output.py's _NO_DISSENT_PREFIXES idiom.
_NO_CRUX_PREFIXES = (
    "none",
    "n/a",
    "no empirical",
    "not empirical",
    "no checkable",
    "no factual",
    "there is no",
)

# Hard bound on the injected block. Round-2 prompts already carry the full anonymized
# Round-1 block; unbounded evidence would re-bill every panelist for a runaway retrieval.
_MAX_EVIDENCE_CHARS = 4000
_MAX_SOURCES_LISTED = 5


def _parse_crux(text: str) -> str | None:
    """Return the crux claim stated under a ``## Crux`` heading, or None.

    None means "no empirical crux" — a valid success, not a parse failure.
    """
    if not text or not text.strip():
        return None

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
                return None
            claim = stripped.strip("*_` ").strip()
            if not claim:
                return None
            if claim.lower().startswith(_NO_CRUX_PREFIXES):
                return None
            return claim
        return None
    return None


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
    return "\n".join(parts)


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

        claim = _parse_crux(response.content or "")
        if claim is None:
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
