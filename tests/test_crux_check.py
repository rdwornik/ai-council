"""Tests for src/ai_council/crux_check.py.

The contract this file guards (#18):
  - three outcomes: grounded | no_empirical_crux | retrieval_unavailable
  - no_empirical_crux is a VALID SUCCESS and must NEVER trigger retrieval
  - retrieval failure DEGRADES, never raises
  - the service sees only the ALREADY-ANONYMIZED block (ADR-03)
"""

from unittest.mock import AsyncMock

import pytest

from ai_council.crux_check import (
    CruxCheckService,
    ParseState,
    _parse_crux,
    build_crux_check_service,
)
from ai_council.models import CruxStatus, ModelResponse
from ai_council.providers.base import ProviderError
from ai_council.research.models import MergedResearchReport, ResearchResult, Source
from config.config_loader import AppConfig, CruxCheckConfig
from tests.conftest import MockProvider


def _crux_cfg(**overrides) -> CruxCheckConfig:
    base = dict(
        providers=["perplexity"],
        budget_sec=90.0,
        injection_header="--- Evidence check ---",
        extraction_prompt="Q: {question}\n\n{anon_block}",
    )
    base.update(overrides)
    return CruxCheckConfig(**base)


def _report(content: str = "Retrieved evidence.", n_sources: int = 2) -> MergedResearchReport:
    results = [
        ResearchResult(
            provider="perplexity",
            query="q",
            content=content,
            sources=[Source(title=f"T{i}", url=f"https://e.test/{i}") for i in range(n_sources)],
        )
    ]
    return MergedResearchReport(
        query="q",
        results=results,
        merged_report=content,
        summary_2500=content,
        total_sources=n_sources,
    )


def _service(
    config: AppConfig,
    extractor_content: str = "## Crux\nDeploys fail more on Fridays.",
    executor=None,
    extractor: MockProvider | None = None,
) -> CruxCheckService:
    prov = extractor or MockProvider("openai", extractor_content)
    if executor is None:
        executor = AsyncMock(return_value=_report())
    return CruxCheckService(prov, config, executor=executor)


@pytest.fixture
def config(sample_app_config: AppConfig) -> AppConfig:
    sample_app_config.crux_check = _crux_cfg()
    return sample_app_config


# --------------------------------------------------------------------------- parsing


class TestParseCrux:
    def test_extracts_claim_from_crux_heading(self):
        assert _parse_crux("## Crux\nDeploys fail more on Fridays.") == (
            ParseState.CLAIM,
            "Deploys fail more on Fridays.",
        )

    def test_none_sentinel_is_no_crux(self):
        assert _parse_crux("## Crux\nNONE")[0] is ParseState.NO_CRUX

    def test_explicit_crux_absence_prose_is_no_crux(self):
        assert _parse_crux("## Crux\nNo empirical disagreement.")[0] is ParseState.NO_CRUX

    def test_heading_match_is_case_insensitive(self):
        assert _parse_crux("## CRUX\nX happens.") == (ParseState.CLAIM, "X happens.")
        assert _parse_crux("## crux\nX happens.") == (ParseState.CLAIM, "X happens.")

    def test_takes_first_nonempty_line_only(self):
        text = "## Crux\n\nFirst claim.\nSecond line ignored."
        assert _parse_crux(text) == (ParseState.CLAIM, "First claim.")

    def test_ignores_preamble_before_heading(self):
        text = "Here is my analysis.\n\n## Crux\nThe real claim."
        assert _parse_crux(text) == (ParseState.CLAIM, "The real claim.")

    def test_long_claim_is_truncated(self):
        claim = "x" * 900
        state, parsed = _parse_crux(f"## Crux\n{claim}")
        assert state is ParseState.CLAIM
        assert len(parsed) < 500

    # --- terra HIGH-1: malformed must NOT masquerade as a no-crux success ---

    def test_missing_heading_is_malformed_not_no_crux(self):
        """A refusal is an extraction FAILURE — reporting it as no-crux would claim we
        checked and found nothing, when we never found out."""
        assert _parse_crux("I could not identify anything checkable.")[0] is (
            ParseState.MALFORMED
        )

    def test_empty_response_is_malformed(self):
        assert _parse_crux("")[0] is ParseState.MALFORMED
        assert _parse_crux("   \n\n  ")[0] is ParseState.MALFORMED

    def test_heading_with_empty_body_is_malformed(self):
        assert _parse_crux("## Crux\n\n## Notes\nnot the crux")[0] is ParseState.MALFORMED

    def test_truncated_response_is_malformed(self):
        assert _parse_crux("## Cru")[0] is ParseState.MALFORMED

    # --- terra HIGH-1: a negative claim is still a CLAIM ---

    def test_negative_empirical_claim_is_a_claim_not_no_crux(self):
        """"There is no significant difference" is textbook checkable — a generic negation
        prefix used to discard it as no-crux, silently skipping retrieval."""
        text = "## Crux\nThere is no statistically significant difference between A and B."
        state, claim = _parse_crux(text)
        assert state is ParseState.CLAIM
        assert claim.startswith("There is no statistically")

    def test_no_evidence_claim_is_a_claim(self):
        text = "## Crux\nNo published benchmark shows X outperforming Y."
        assert _parse_crux(text)[0] is ParseState.CLAIM

    # --- terra pass-2: "none" as a PREFIX swallowed valid claims ---

    def test_nonetheless_is_a_claim_not_the_none_sentinel(self):
        """startswith("none") matched "Nonetheless" — a sentinel must match exactly."""
        text = "## Crux\nNonetheless, deployments fail more often on Fridays."
        state, claim = _parse_crux(text)
        assert state is ParseState.CLAIM
        assert claim.startswith("Nonetheless")

    def test_none_of_the_benchmarks_is_a_claim(self):
        text = "## Crux\nNone of the benchmarks met the target."
        state, claim = _parse_crux(text)
        assert state is ParseState.CLAIM
        assert claim.startswith("None of the")

    def test_bare_none_with_punctuation_is_still_the_sentinel(self):
        for variant in ("NONE", "None.", "none", "n/a", "N/A."):
            assert _parse_crux(f"## Crux\n{variant}")[0] is ParseState.NO_CRUX, variant

    # --- terra pass-2: refusals under a valid heading are failures, not claims ---

    def test_headed_refusal_is_malformed_not_a_claim(self):
        """Otherwise the refusal text is sent to retrieval and a hit reads as GROUNDED."""
        text = "## Crux\nI cannot determine a crux because the input is incomplete."
        assert _parse_crux(text)[0] is ParseState.MALFORMED

    def test_headed_uncertainty_is_malformed(self):
        for refusal in (
            "I'm sorry, I can't help with this.",
            "There is insufficient information to identify a crux.",
            "As an AI, I am unable to assess this.",
        ):
            assert _parse_crux(f"## Crux\n{refusal}")[0] is ParseState.MALFORMED, refusal


# ------------------------------------------------------------------------- outcomes


class TestCruxStatuses:
    async def test_grounded_produces_evidence_block(self, config):
        svc = _service(config)
        art = await svc.check("Should we deploy on Fridays?", "--- Proposal A ---\nyes")
        assert art.status is CruxStatus.GROUNDED
        assert art.crux_claim == "Deploys fail more on Fridays."
        assert "Retrieved evidence." in art.evidence_block
        assert config.crux_check.injection_header in art.evidence_block
        assert art.sources_count == 2

    async def test_no_empirical_crux_never_invokes_retrieval(self, config):
        """The 'never fabricate retrieval' guarantee, asserted structurally."""
        executor = AsyncMock(return_value=_report())
        svc = _service(config, extractor_content="## Crux\nNONE", executor=executor)
        art = await svc.check("Which font is nicer?", "--- Proposal A ---\nserif")
        assert art.status is CruxStatus.NO_EMPIRICAL_CRUX
        assert art.evidence_block == ""
        executor.assert_not_awaited()

    async def test_no_empirical_crux_still_records_call_metrics(self, config):
        """The extraction call happened, so its cost must still be booked."""
        svc = _service(config, extractor_content="## Crux\nNONE")
        art = await svc.check("q", "block")
        assert art.call_metrics is not None
        assert art.call_metrics.round_number == -1

    async def test_retrieval_unavailable_when_executor_returns_none(self, config):
        svc = _service(config, executor=AsyncMock(return_value=None))
        art = await svc.check("q", "block")
        assert art.status is CruxStatus.RETRIEVAL_UNAVAILABLE
        assert art.evidence_block == ""
        assert art.detail

    async def test_retrieval_unavailable_when_all_providers_error(self, config):
        errored = MergedResearchReport(
            query="q",
            results=[ResearchResult(provider="perplexity", query="q", content="", error="down")],
            merged_report="",
            summary_2500="",
        )
        svc = _service(config, executor=AsyncMock(return_value=errored))
        art = await svc.check("q", "block")
        assert art.status is CruxStatus.RETRIEVAL_UNAVAILABLE

    async def test_retrieval_unavailable_on_executor_exception(self, config):
        """A crashing executor must degrade, never propagate into the debate."""
        svc = _service(config, executor=AsyncMock(side_effect=RuntimeError("boom")))
        art = await svc.check("q", "block")
        assert art.status is CruxStatus.RETRIEVAL_UNAVAILABLE

    async def test_extractor_provider_error_returns_retrieval_unavailable(self, config):
        prov = MockProvider("openai")
        prov.generate = AsyncMock(side_effect=ProviderError("openai", "rate limited"))
        svc = _service(config, extractor=prov)
        art = await svc.check("q", "block")
        assert art.status is CruxStatus.RETRIEVAL_UNAVAILABLE
        assert art.call_metrics is None  # the call never produced a response

    async def test_call_metrics_uses_round_number_minus_one(self, config):
        """0 is taken by synthesis, 1..n by rounds; -1 is the out-of-band sentinel."""
        svc = _service(config)
        art = await svc.check("q", "block")
        assert art.call_metrics is not None
        assert art.call_metrics.round_number == -1

    async def test_budget_and_providers_forwarded_to_executor(self, config):
        executor = AsyncMock(return_value=_report())
        svc = _service(config, executor=executor)
        await svc.check("q", "block")
        kwargs = executor.await_args.kwargs
        assert kwargs["provider_names"] == ["perplexity"]
        assert kwargs["budget_sec"] == 90.0

    async def test_retrieval_queries_the_crux_not_the_question(self, config):
        """The whole point: we check the extracted claim, not the original question."""
        executor = AsyncMock(return_value=_report())
        svc = _service(config, executor=executor)
        await svc.check("Should we deploy on Fridays?", "block")
        assert executor.await_args.args[0] == "Deploys fail more on Fridays."


# -------------------------------------------------------------------- anonymization


class TestAnonymization:
    async def test_evidence_block_contains_no_panel_provider_names(self, config):
        """ADR-03: the artifact must carry zero provider/model attribution.

        The report is deliberately poisoned: merged_report carries the per-provider
        "## gemini"-style headers that merge_results really produces, and several research
        provider names (gemini, grok, openai) collide with PANEL model names. If the
        implementation ever switches to merged_report/summary_2500, this fails.
        """
        poisoned = MergedResearchReport(
            query="q",
            results=[
                ResearchResult(
                    provider="gemini",
                    query="q",
                    content="Fridays show a 12% higher incident rate.",
                    sources=[Source(title="T", url="https://e.test/1")],
                )
            ],
            merged_report="## gemini\nFridays show...\n\n## grok\nAlso openai and claude say...",
            summary_2500="## gemini\nsummary mentioning claude and deepseek",
            total_sources=1,
        )
        svc = _service(config, executor=AsyncMock(return_value=poisoned))
        art = await svc.check("q", "--- Proposal A ---\ncontent")

        assert art.status is CruxStatus.GROUNDED
        assert "Fridays show a 12% higher incident rate." in art.evidence_block
        for name in ("claude", "gemini", "openai", "grok", "deepseek"):
            assert name not in art.evidence_block.lower(), f"{name} leaked into the artifact"

    async def test_check_receives_only_the_anonymized_block(self, config):
        """The signature takes str, not list[ModelResponse] — ADR-03 by construction."""
        import inspect

        sig = inspect.signature(CruxCheckService.check)
        assert list(sig.parameters) == ["self", "question_text", "anon_block"]
        assert sig.parameters["anon_block"].annotation in (str, "str")

    async def test_extraction_prompt_embeds_the_anon_block(self, config):
        prov = MockProvider("openai", "## Crux\nX.")
        svc = _service(config, extractor=prov)
        await svc.check("My question", "--- Proposal A ---\nsome claim")
        sent = prov.generate.await_args.args[0]
        assert "--- Proposal A ---" in sent
        assert "My question" in sent
        # A raw ModelResponse would leak the provider name; the block must not.
        assert not isinstance(sent, ModelResponse)


# ------------------------------------------------------------------------- builder


class TestBuildCruxCheckService:
    def test_returns_none_when_config_absent(self, sample_app_config):
        sample_app_config.crux_check = None
        assert build_crux_check_service(sample_app_config, MockProvider("openai")) is None

    def test_returns_none_when_no_providers_configured(self, sample_app_config):
        sample_app_config.crux_check = _crux_cfg(providers=[])
        assert build_crux_check_service(sample_app_config, MockProvider("openai")) is None

    def test_returns_none_when_extraction_prompt_missing(self, sample_app_config):
        sample_app_config.crux_check = _crux_cfg(extraction_prompt="")
        assert build_crux_check_service(sample_app_config, MockProvider("openai")) is None

    def test_builds_service_when_configured(self, config):
        svc = build_crux_check_service(config, MockProvider("openai"))
        assert isinstance(svc, CruxCheckService)
