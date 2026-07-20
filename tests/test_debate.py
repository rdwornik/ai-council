"""Tests for src/debate.py."""

import logging
from unittest.mock import AsyncMock

import pytest

from ai_council.debate import _anonymize_responses, run_debate
from ai_council.models import (
    CruxArtifact,
    CruxStatus,
    DebateOutcome,
    ModelResponse,
    Round,
)
from ai_council.providers.base import AIProvider, ProviderError, _Parsed
from config.config_loader import ModeConfig, ModelConfig
from tests.conftest import MockProvider


def test_anonymize_responses_uses_proposal_labels():
    responses = [
        ModelResponse("gemini", "gemini-3.1-pro-preview", 1, "Use YAML.", 1.0, 10),
        ModelResponse("claude", "claude-opus-4-6", 1, "Use JSON.", 1.2, 15),
    ]
    block, mapping = _anonymize_responses(responses)
    assert "--- Proposal A ---" in block
    assert "--- Proposal B ---" in block
    assert len(mapping) == 2
    assert set(mapping.keys()) == {"A", "B"}


def test_anonymize_responses_hides_model_names():
    responses = [
        ModelResponse("gemini", "gemini-3.1-pro-preview", 1, "Prefer YAML.", 1.0, 10),
        ModelResponse("claude", "claude-opus-4-6", 1, "Prefer JSON.", 1.2, 15),
    ]
    block, mapping = _anonymize_responses(responses)
    # Provider names and model strings must not appear in the anonymized block
    assert "gemini" not in block
    assert "claude" not in block
    assert "gemini-3.1-pro-preview" not in block
    assert "claude-opus-4-6" not in block
    # Content still present
    assert "Prefer YAML." in block or "Prefer JSON." in block


def test_anonymize_responses_mapping_covers_all_providers():
    responses = [
        ModelResponse("provider_a", "model-a", 1, "A says yes.", 0.5, 5),
        ModelResponse("provider_b", "model-b", 1, "B says no.", 0.5, 5),
        ModelResponse("provider_c", "model-c", 1, "C says maybe.", 0.5, 5),
    ]
    block, mapping = _anonymize_responses(responses)
    assert set(mapping.values()) == {"provider_a", "provider_b", "provider_c"}
    assert len(mapping) == 3


async def test_run_debate_single_round(
    two_mock_providers, sample_prompts_config, sample_question
):
    outcome = await run_debate(
        question=sample_question,
        providers=two_mock_providers,
        prompts=sample_prompts_config,
        num_rounds=1,
    )
    assert isinstance(outcome, DebateOutcome)
    assert len(outcome.rounds) == 1
    assert outcome.rounds[0].number == 1
    assert len(outcome.rounds[0].responses) == 2


async def test_run_debate_two_rounds(
    two_mock_providers, sample_prompts_config, sample_question
):
    for p in two_mock_providers:
        p.generate = AsyncMock(
            side_effect=[
                ModelResponse(
                    p.name(), "mock-model", 1, f"Round 1 from {p.name()}", 0.1, 5
                ),
                ModelResponse(
                    p.name(), "mock-model", 2, f"Round 2 from {p.name()}", 0.1, 5
                ),
            ]
        )

    outcome = await run_debate(
        question=sample_question,
        providers=two_mock_providers,
        prompts=sample_prompts_config,
        num_rounds=2,
    )
    assert len(outcome.rounds) == 2
    assert outcome.rounds[1].number == 2


async def test_run_debate_injects_persona_in_round1(
    sample_prompts_config, sample_question
):
    """Persona text should appear in the prompt passed to the provider on round 1."""
    from ai_council.providers.base import AIProvider, _Parsed

    captured_prompts: list[str] = []

    class CapturingProvider(AIProvider):
        def __init__(self, provider_name: str) -> None:
            self._name = provider_name
            self._config = ModelConfig(
                name=provider_name, sdk="mock", model="mock-model",
                api_key_env="K", timeout_sec=30, max_tokens=100,
            )

        def name(self) -> str:
            return self._name

        def model_string(self) -> str:
            return "mock-model"

        async def generate(
            self, prompt: str, round_number: int, *, timeout: float | None = None
        ) -> ModelResponse:
            captured_prompts.append(prompt)
            return ModelResponse(
                self._name, "mock-model", round_number, "response", 0.1, 5
            )

        # A1 ABC hooks — never called (generate is overridden above); satisfy the ABC.
        async def _invoke(self, prompt: str) -> object:
            return None

        def _parse(self, raw: object) -> _Parsed:
            return _Parsed("response")

    provider1 = CapturingProvider("mock")
    provider2 = CapturingProvider("mock2")

    outcome = await run_debate(
        question=sample_question,
        providers=[provider1, provider2],
        prompts=sample_prompts_config,
        num_rounds=1,
    )

    # The persona for "mock" is "Be a mock architect." (from sample_prompts_config fixture)
    assert any("Be a mock architect." in p for p in captured_prompts)
    assert isinstance(outcome, DebateOutcome)


async def test_run_debate_critique_uses_anonymized_placeholder(
    two_mock_providers, sample_prompts_config, sample_question
):
    """Round 2 prompts should use previous_responses_anonymized, not provider names."""
    captured_prompts: list[str] = []

    for p in two_mock_providers:
        original_name = p.name()
        p.generate = AsyncMock(
            side_effect=[
                ModelResponse(
                    original_name,
                    "mock-model",
                    1,
                    f"Round 1 from {original_name}",
                    0.1,
                    5,
                ),
                ModelResponse(
                    original_name,
                    "mock-model",
                    2,
                    f"Round 2 from {original_name}",
                    0.1,
                    5,
                ),
            ]
        )

    # Patch generate to capture prompts on round 2
    for p in two_mock_providers:
        original_side_effect = p.generate.side_effect

        async def capturing_generate(prompt, round_number, _orig=original_side_effect):
            if round_number == 2:
                captured_prompts.append(prompt)
            return await AsyncMock(side_effect=_orig)(prompt, round_number)

        p.generate = AsyncMock(side_effect=capturing_generate)

    # Re-set side effects
    for p in two_mock_providers:
        original_name = p.name()
        p.generate = AsyncMock(
            side_effect=[
                ModelResponse(
                    original_name,
                    "mock-model",
                    1,
                    f"Round 1 from {original_name}",
                    0.1,
                    5,
                ),
                ModelResponse(
                    original_name,
                    "mock-model",
                    2,
                    f"Round 2 from {original_name}",
                    0.1,
                    5,
                ),
            ]
        )

    outcome = await run_debate(
        question=sample_question,
        providers=two_mock_providers,
        prompts=sample_prompts_config,
        num_rounds=2,
    )
    assert len(outcome.rounds) == 2
    # Ensure round 2 actually ran
    assert outcome.rounds[1].number == 2


async def test_run_debate_on_round_complete_callback(
    two_mock_providers, sample_prompts_config, sample_question
):
    completed = []

    def callback(rnd: Round) -> None:
        completed.append(rnd.number)

    outcome = await run_debate(
        question=sample_question,
        providers=two_mock_providers,
        prompts=sample_prompts_config,
        num_rounds=2,
        on_round_complete=callback,
    )
    assert completed == [1, 2]
    assert isinstance(outcome, DebateOutcome)


async def test_run_debate_skips_failed_provider(sample_prompts_config, sample_question):
    good = MockProvider("good", "Good response")
    bad = MockProvider("bad", "")
    bad.generate = AsyncMock(side_effect=ProviderError("bad", "API error"))

    outcome = await run_debate(
        question=sample_question,
        providers=[good, bad],
        prompts=sample_prompts_config,
        num_rounds=1,
    )
    assert len(outcome.rounds) == 1
    assert len(outcome.rounds[0].responses) == 1
    assert outcome.rounds[0].responses[0].provider == "good"


async def test_run_debate_raises_if_all_fail(sample_prompts_config, sample_question):
    p1 = MockProvider("p1")
    p2 = MockProvider("p2")
    p1.generate = AsyncMock(side_effect=ProviderError("p1", "fail"))
    p2.generate = AsyncMock(side_effect=ProviderError("p2", "fail"))

    with pytest.raises(RuntimeError, match="All providers failed"):
        await run_debate(
            question=sample_question,
            providers=[p1, p2],
            prompts=sample_prompts_config,
            num_rounds=1,
        )


# --- Retry logic tests ---


def _make_mock_config(timeout_sec: int = 10) -> ModelConfig:
    return ModelConfig(
        name="slow",
        sdk="test",
        model="m",
        api_key_env="K",
        timeout_sec=timeout_sec,
        max_tokens=100,
    )


async def test_retry_succeeds_on_second_attempt(sample_prompts_config, sample_question):
    """Provider that times out once but succeeds on retry is included in results."""
    good = MockProvider("good", "Good response")
    slow = MockProvider("slow", "Slow but eventual response")
    slow._config = _make_mock_config(10)

    call_count = 0

    async def timeout_then_succeed(
        prompt: str, round_number: int, *, timeout: float | None = None
    ) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ProviderError("slow", "Request timed out after 10s")
        return ModelResponse(
            "slow", "m", round_number, "Slow but eventual response", 0.1, 5
        )

    slow.generate = AsyncMock(side_effect=timeout_then_succeed)

    outcome = await run_debate(
        question=sample_question,
        providers=[good, slow],
        prompts=sample_prompts_config,
        num_rounds=1,
    )
    assert len(outcome.rounds[0].responses) == 2
    assert call_count == 2  # initial attempt + one retry


async def test_retry_excluded_after_second_timeout(
    sample_prompts_config, sample_question
):
    """Provider that times out on both attempts is excluded from results."""
    good = MockProvider("good", "Good response")
    slow = MockProvider("slow", "")
    slow._config = _make_mock_config(10)
    slow.generate = AsyncMock(
        side_effect=ProviderError("slow", "Request timed out after 10s")
    )

    outcome = await run_debate(
        question=sample_question,
        providers=[good, slow],
        prompts=sample_prompts_config,
        num_rounds=1,
    )
    assert len(outcome.rounds[0].responses) == 1
    assert outcome.rounds[0].responses[0].provider == "good"


async def test_retry_uses_15x_timeout(sample_prompts_config, sample_question):
    """On retry, the timeout kwarg grows to 1.5x — WITHOUT mutating provider config (A3)."""
    good = MockProvider("good", "Good response")
    slow = MockProvider("slow", "Slow response")
    slow._config = _make_mock_config(100)

    observed_timeouts: list[float | None] = []

    async def capture_timeout(
        prompt: str, round_number: int, *, timeout: float | None = None
    ) -> ModelResponse:
        observed_timeouts.append(timeout)
        if len(observed_timeouts) == 1:
            raise ProviderError("slow", "Request timed out after 100s")
        return ModelResponse("slow", "m", round_number, "Slow response", 0.1, 5)

    slow.generate = AsyncMock(side_effect=capture_timeout)

    await run_debate(
        question=sample_question,
        providers=[good, slow],
        prompts=sample_prompts_config,
        num_rounds=1,
    )
    assert observed_timeouts[0] == 100  # base timeout on first attempt (1.5**0)
    assert observed_timeouts[1] == 150  # 1.5x on retry, via the kwarg
    assert slow._config.timeout_sec == 100  # never mutated — the reach-through is gone


async def test_was_retry_set_on_retry_response(sample_prompts_config, sample_question):
    """Response from a retry attempt has was_retry=True; first-attempt responses have was_retry=False."""
    good = MockProvider("good", "Good response")
    slow = MockProvider("slow", "Recovered response")
    slow._config = _make_mock_config(10)

    call_count = 0

    async def timeout_then_succeed(
        prompt: str, round_number: int, *, timeout: float | None = None
    ) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ProviderError("slow", "Request timed out after 10s")
        return ModelResponse("slow", "m", round_number, "Recovered response", 0.1, 5)

    slow.generate = AsyncMock(side_effect=timeout_then_succeed)

    outcome = await run_debate(
        question=sample_question,
        providers=[good, slow],
        prompts=sample_prompts_config,
        num_rounds=1,
    )

    good_resp = next(r for r in outcome.rounds[0].responses if r.provider == "good")
    slow_resp = next(r for r in outcome.rounds[0].responses if r.provider == "slow")
    assert good_resp.was_retry is False
    assert slow_resp.was_retry is True


# --- Quality gate tests ---


async def test_quality_gate_warns_when_too_few_respond(
    sample_prompts_config, sample_question, caplog
):
    """Quality gate fires when fewer than 3 models respond in Round 1 on a 4-model panel."""
    p1 = MockProvider("p1", "Response 1")
    p2 = MockProvider("p2", "Response 2")
    p3 = MockProvider("p3", "")
    p4 = MockProvider("p4", "")
    p3.generate = AsyncMock(side_effect=ProviderError("p3", "fail"))
    p4.generate = AsyncMock(side_effect=ProviderError("p4", "fail"))

    with caplog.at_level(logging.WARNING):
        outcome = await run_debate(
            question=sample_question,
            providers=[p1, p2, p3, p4],
            prompts=sample_prompts_config,
            num_rounds=1,
        )

    assert isinstance(outcome, DebateOutcome)
    assert any("Only 2/4" in msg for msg in caplog.messages)
    assert any("Debate quality is degraded" in msg for msg in caplog.messages)


async def test_quality_gate_silent_for_small_panels(
    sample_prompts_config, sample_question, caplog
):
    """Quality gate does not fire when panel has fewer than 3 providers."""
    p1 = MockProvider("p1", "Response 1")
    p2 = MockProvider("p2", "Response 2")

    with caplog.at_level(logging.WARNING):
        await run_debate(
            question=sample_question,
            providers=[p1, p2],
            prompts=sample_prompts_config,
            num_rounds=1,
        )

    assert not any("Debate quality is degraded" in msg for msg in caplog.messages)


async def test_quality_gate_silent_when_enough_respond(
    sample_prompts_config, sample_question, caplog
):
    """Quality gate does not fire when >= 3 providers succeed."""
    providers = [MockProvider(f"p{i}", f"Response {i}") for i in range(1, 5)]

    with caplog.at_level(logging.WARNING):
        await run_debate(
            question=sample_question,
            providers=providers,
            prompts=sample_prompts_config,
            num_rounds=1,
        )

    assert not any("Debate quality is degraded" in msg for msg in caplog.messages)


# --- DebateOutcome degradation tests ---


async def test_run_debate_returns_outcome_with_statuses(
    two_mock_providers, sample_prompts_config, sample_question
):
    """Successful run populates provider_statuses as 'ok' for all providers."""
    outcome = await run_debate(
        question=sample_question,
        providers=two_mock_providers,
        prompts=sample_prompts_config,
        num_rounds=1,
    )
    assert outcome.degraded is False
    assert outcome.degradation_summary is None
    for p in two_mock_providers:
        assert outcome.provider_statuses[p.name()] == "ok"


async def test_run_debate_failed_provider_marked_in_statuses(
    sample_prompts_config, sample_question
):
    """A provider that always fails is marked 'failed' in provider_statuses."""
    good = MockProvider("good", "Good response")
    bad = MockProvider("bad", "")
    bad.generate = AsyncMock(side_effect=ProviderError("bad", "API error"))

    outcome = await run_debate(
        question=sample_question,
        providers=[good, bad],
        prompts=sample_prompts_config,
        num_rounds=1,
    )
    assert outcome.provider_statuses["good"] == "ok"
    assert outcome.provider_statuses["bad"] == "failed"


async def test_run_debate_round2_all_fail_returns_partial(
    sample_prompts_config, sample_question
):
    """Round 2+ all-fail returns DebateOutcome with degraded=True instead of raising."""
    p1 = MockProvider("p1", "Round 1 response")
    p2 = MockProvider("p2", "Round 1 response")

    async def p1_generate(
        prompt: str, round_number: int, *, timeout: float | None = None
    ) -> ModelResponse:
        if round_number == 1:
            return ModelResponse("p1", "mock-model", round_number, "Round 1 from p1", 0.1, 5)
        raise ProviderError("p1", "fail round 2")

    async def p2_generate(
        prompt: str, round_number: int, *, timeout: float | None = None
    ) -> ModelResponse:
        if round_number == 1:
            return ModelResponse("p2", "mock-model", round_number, "Round 1 from p2", 0.1, 5)
        raise ProviderError("p2", "fail round 2")

    p1.generate = AsyncMock(side_effect=p1_generate)
    p2.generate = AsyncMock(side_effect=p2_generate)

    outcome = await run_debate(
        question=sample_question,
        providers=[p1, p2],
        prompts=sample_prompts_config,
        num_rounds=2,
    )

    assert outcome.degraded is True
    assert outcome.degradation_summary is not None
    assert "round 2" in outcome.degradation_summary.lower()
    assert len(outcome.rounds) == 1  # only round 1 completed
    assert outcome.rounds[0].number == 1


# --------------------------------------------------------------- #18 bounded crux check


class _StubCruxChecker:
    """Records what the debate hands the crux service, and what it hands back."""

    def __init__(self, artifact: CruxArtifact | None = None) -> None:
        self.artifact = artifact or CruxArtifact(
            status=CruxStatus.GROUNDED,
            crux_claim="Deploys fail more on Fridays.",
            evidence_block="--- Evidence ---\nIncident rate is 12% higher.",
        )
        self.calls: list[tuple[str, str]] = []

    async def check(self, question_text: str, anon_block: str) -> CruxArtifact:
        self.calls.append((question_text, anon_block))
        return self.artifact


def _capturing_providers(names: list[str], captured: list[str]) -> list[AIProvider]:
    """Providers that record every Round-2+ prompt they are given."""

    class CapturingProvider(AIProvider):
        def __init__(self, provider_name: str) -> None:
            self._name = provider_name
            self._config = ModelConfig(
                name=provider_name, sdk="mock", model="mock-model",
                api_key_env="K", timeout_sec=30, max_tokens=100,
            )

        def name(self) -> str:
            return self._name

        def model_string(self) -> str:
            return "mock-model"

        async def generate(
            self, prompt: str, round_number: int, *, timeout: float | None = None
        ) -> ModelResponse:
            if round_number >= 2:
                captured.append(prompt)
            return ModelResponse(self._name, "mock-model", round_number, "resp", 0.1, 5)

        async def _invoke(self, prompt: str) -> object:
            return None

        def _parse(self, raw: object) -> _Parsed:
            return _Parsed("resp")

    return [CapturingProvider(n) for n in names]


async def test_crux_check_called_once_before_round_two(sample_question, sample_prompts_config):
    checker = _StubCruxChecker()
    providers = [MockProvider("a"), MockProvider("b")]
    await run_debate(
        sample_question, providers, sample_prompts_config, 2, crux_check=checker
    )
    assert len(checker.calls) == 1


async def test_crux_check_not_called_when_num_rounds_is_one(
    sample_question, sample_prompts_config
):
    """A one-round debate has no Round 2 to inject into — spending the call would be waste."""
    checker = _StubCruxChecker()
    await run_debate(
        sample_question, [MockProvider("a")], sample_prompts_config, 1, crux_check=checker
    )
    assert checker.calls == []


async def test_crux_check_called_once_even_across_three_rounds(
    sample_question, sample_prompts_config
):
    """The contract is ONE call between R1 and R2; rounds 3+ reuse the stored artifact."""
    checker = _StubCruxChecker()
    captured: list[str] = []
    providers = _capturing_providers(["a", "b"], captured)
    await run_debate(
        sample_question, providers, sample_prompts_config, 3, crux_check=checker
    )
    assert len(checker.calls) == 1
    # Rounds 2 and 3, two providers each = 4 prompts, all carrying the evidence.
    assert len(captured) == 4
    assert all("Incident rate is 12% higher." in p for p in captured)


async def test_crux_service_receives_anonymized_block_not_responses(
    sample_question, sample_prompts_config
):
    """ADR-03: what reaches the service is the shuffled Proposal-labelled block."""
    checker = _StubCruxChecker()
    await run_debate(
        sample_question,
        [MockProvider("gemini"), MockProvider("claude")],
        sample_prompts_config,
        2,
        crux_check=checker,
    )
    question_text, anon_block = checker.calls[0]
    assert question_text == sample_question.text
    assert "--- Proposal A ---" in anon_block
    assert "gemini" not in anon_block
    assert "claude" not in anon_block


async def test_crux_evidence_appears_in_round2_prompt_pick_mode(
    sample_question, sample_prompts_config
):
    captured: list[str] = []
    providers = _capturing_providers(["a", "b"], captured)
    checker = _StubCruxChecker()
    await run_debate(
        sample_question, providers, sample_prompts_config, 2, crux_check=checker
    )
    assert len(captured) == 2
    for prompt in captured:
        assert "Incident rate is 12% higher." in prompt
    # All panelists get the SAME evidence — one canonical artifact, not per-seat variants.
    assert captured[0].split("--- Evidence ---")[1] == captured[1].split("--- Evidence ---")[1]


async def test_crux_evidence_appears_in_round2_prompt_ideas_mode(
    sample_question, sample_prompts_config
):
    # uses_existing_prompts is a computed property: a non-empty round1_instruction is what
    # takes this off the pick-mode branch and onto the ideas/judge assembly branch.
    mode = ModeConfig(
        description="ideas",
        emoji="",
        aliases=[],
        default=False,
        max_rounds=3,
        token_budget=1000,
        round1_instruction="Give ideas.",
        round2_instruction="Now converge.",
    )
    assert mode.uses_existing_prompts is False
    captured: list[str] = []
    providers = _capturing_providers(["a"], captured)
    checker = _StubCruxChecker()
    await run_debate(
        sample_question,
        providers,
        sample_prompts_config,
        2,
        mode_config=mode,
        crux_check=checker,
    )
    prompt = captured[0]
    assert "Incident rate is 12% higher." in prompt
    # Evidence sits AFTER the anonymized block and BEFORE the round-2 instruction.
    assert prompt.index("--- Evidence ---") < prompt.index("Now converge.")


async def test_no_empirical_crux_leaves_round2_prompt_byte_identical(
    sample_question, sample_prompts_config
):
    """The valid-success path must not perturb the prompt at all."""
    baseline: list[str] = []
    await run_debate(
        sample_question, _capturing_providers(["a"], baseline), sample_prompts_config, 2
    )

    with_checker: list[str] = []
    checker = _StubCruxChecker(
        CruxArtifact(status=CruxStatus.NO_EMPIRICAL_CRUX, evidence_block="")
    )
    await run_debate(
        sample_question,
        _capturing_providers(["a"], with_checker),
        sample_prompts_config,
        2,
        crux_check=checker,
    )
    assert with_checker[0] == baseline[0]


async def test_retrieval_unavailable_does_not_abort_debate(
    sample_question, sample_prompts_config
):
    checker = _StubCruxChecker(
        CruxArtifact(status=CruxStatus.RETRIEVAL_UNAVAILABLE, detail="no providers")
    )
    outcome = await run_debate(
        sample_question,
        [MockProvider("a"), MockProvider("b")],
        sample_prompts_config,
        2,
        crux_check=checker,
    )
    assert len(outcome.rounds) == 2
    # Degradation here is the RESEARCH lane's, not the debate's — it must NOT leak into the
    # verdict package's degradation block (contract_version 1.0 is frozen at Phase A).
    assert outcome.degraded is False
    assert outcome.crux is not None
    assert outcome.crux.status is CruxStatus.RETRIEVAL_UNAVAILABLE


async def test_crux_service_exception_does_not_abort_debate(
    sample_question, sample_prompts_config
):
    """Defence in depth: even a service that breaks its own no-raise contract is contained."""

    class _ExplodingChecker:
        async def check(self, question_text: str, anon_block: str) -> CruxArtifact:
            raise RuntimeError("service bug")

    outcome = await run_debate(
        sample_question,
        [MockProvider("a")],
        sample_prompts_config,
        2,
        crux_check=_ExplodingChecker(),
    )
    assert len(outcome.rounds) == 2
    # terra HIGH-3: the failure must be RECORDED as retrieval_unavailable, not collapsed to
    # None — None is the "no service injected" state, and conflating them erases the third
    # outcome from DebateOutcome and from the console line the orchestrator prints.
    assert outcome.crux is not None
    assert outcome.crux.status is CruxStatus.RETRIEVAL_UNAVAILABLE
    assert "service bug" in (outcome.crux.detail or "")


async def test_crux_check_none_preserves_existing_behavior(
    sample_question, sample_prompts_config
):
    outcome = await run_debate(
        sample_question, [MockProvider("a"), MockProvider("b")], sample_prompts_config, 2
    )
    assert outcome.crux is None
    assert len(outcome.rounds) == 2


async def test_crux_artifact_attached_to_debate_outcome(
    sample_question, sample_prompts_config
):
    checker = _StubCruxChecker()
    outcome = await run_debate(
        sample_question, [MockProvider("a")], sample_prompts_config, 2, crux_check=checker
    )
    assert outcome.crux is checker.artifact
    assert outcome.crux.status is CruxStatus.GROUNDED
