"""Tests for CouncilRunner and module-level runner helpers."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from ai_council.models import (
    DebateOutcome,
    DebateResult,
    ModelResponse,
    Question,
    Round,
    RunRequest,
)
from ai_council.orchestrator import CouncilRunner
from ai_council.policy import RunPolicy
from ai_council.providers.base import ProviderError
from ai_council.runner import (
    build_all_providers,
    determine_panel,
    exclude_synthesizer_from_panel,
    pick_synthesizer,
)
from config.config_loader import AppConfig, DefaultsConfig, ModelConfig, PromptsConfig
from tests.conftest import MockProvider

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def all_providers():
    return {
        "claude": MockProvider("claude"),
        "gemini": MockProvider("gemini"),
        "deepseek": MockProvider("deepseek"),
        "openai": MockProvider("openai"),
        "grok": MockProvider("grok"),
    }


@pytest.fixture
def two_providers():
    return {
        "claude": MockProvider("claude"),
        "gemini": MockProvider("gemini"),
    }


@pytest.fixture
def multi_model_config(tmp_path):
    """AppConfig with claude + gemini + openai."""
    def _model(name):
        return ModelConfig(
            name=name, sdk="test", model=f"{name}-model",
            api_key_env="TEST_KEY", timeout_sec=60, max_tokens=1000,
        )

    defaults = DefaultsConfig(
        rounds=1,
        max_rounds=3,
        output_dir=tmp_path / "output",
        synthesizer="openai",
        default_panel=["claude", "gemini"],
        full_panel=["claude", "gemini", "openai"],
    )
    prompts = PromptsConfig(
        initial="{persona}\nAnswer: {question}",
        critique="{persona}\nRound {round}. {question}\n{previous_responses_anonymized}",
        synthesis="Question: {question}\n\n{full_transcript}\n\nSynthesize:",
        personas={"claude": "Be Claude.", "gemini": "Be Gemini.", "openai": "Be GPT."},
    )
    return AppConfig(
        defaults=defaults,
        models={"claude": _model("claude"), "gemini": _model("gemini"), "openai": _model("openai")},
        prompts=prompts,
        available_providers={"claude", "gemini", "openai"},
    )


# ---------------------------------------------------------------------------
# determine_panel
# ---------------------------------------------------------------------------


def test_determine_panel_default(multi_model_config):
    panel, mode = determine_panel(multi_model_config, models_arg=None, full_flag=False)
    assert panel == multi_model_config.defaults.default_panel
    assert mode == "default"


def test_determine_panel_full(multi_model_config):
    panel, mode = determine_panel(multi_model_config, models_arg=None, full_flag=True)
    assert panel == multi_model_config.defaults.full_panel
    assert mode == "full"


def test_determine_panel_custom(multi_model_config):
    panel, mode = determine_panel(multi_model_config, models_arg="claude,openai", full_flag=False)
    assert panel == ["claude", "openai"]
    assert mode == "custom"


def test_determine_panel_models_overrides_full(multi_model_config):
    panel, mode = determine_panel(multi_model_config, models_arg="claude,gemini", full_flag=True)
    assert panel == ["claude", "gemini"]
    assert mode == "custom"


# ---------------------------------------------------------------------------
# exclude_synthesizer_from_panel
# ---------------------------------------------------------------------------


def test_exclude_synthesizer_when_panel_has_spare(all_providers):
    result = exclude_synthesizer_from_panel(
        ["claude", "gemini", "openai", "grok"], "openai", all_providers
    )
    assert "openai" not in result
    assert len(result) == 3


def test_exclude_synthesizer_not_in_panel(all_providers):
    panel = ["claude", "gemini", "deepseek"]
    result = exclude_synthesizer_from_panel(panel, "openai", all_providers)
    assert result == panel


def test_exclude_synthesizer_keeps_when_only_two(two_providers):
    panel = ["claude", "gemini"]
    result = exclude_synthesizer_from_panel(panel, "gemini", two_providers)
    assert result == panel  # removing would leave only 1


# ---------------------------------------------------------------------------
# pick_synthesizer
# ---------------------------------------------------------------------------


def test_pick_synthesizer_prefers_non_participant(all_providers):
    synth, is_participant = pick_synthesizer(all_providers, ["claude", "gemini", "deepseek"], "openai")
    assert synth.name() == "openai"
    assert is_participant is False


def test_pick_synthesizer_avoids_panel(all_providers):
    synth, is_participant = pick_synthesizer(all_providers, ["claude", "gemini", "deepseek"], "claude")
    assert synth.name() not in ["claude", "gemini", "deepseek"]
    assert is_participant is False


def test_pick_synthesizer_falls_back_to_participant(two_providers):
    synth, is_participant = pick_synthesizer(two_providers, ["claude", "gemini"], "claude")
    assert is_participant is True
    assert synth.name() == "claude"


def test_pick_synthesizer_fallback_to_first_when_preferred_unavailable(two_providers):
    synth, is_participant = pick_synthesizer(two_providers, ["claude", "gemini"], "openai")
    assert is_participant is True
    assert synth.name() in {"claude", "gemini"}


# ---------------------------------------------------------------------------
# build_all_providers
# ---------------------------------------------------------------------------


def test_build_all_providers_skips_unknown(multi_model_config):
    class FakeProvider:
        def __init__(self, cfg):
            self._name = cfg.name
        def name(self): return self._name

    provider_classes = {"claude": FakeProvider, "gemini": FakeProvider}
    # openai is in config.available_providers but not in provider_classes — should be skipped
    providers = build_all_providers(multi_model_config, provider_classes)
    assert "claude" in providers
    assert "gemini" in providers
    assert "openai" not in providers


def test_build_all_providers_handles_instantiation_error(multi_model_config):
    class BrokenProvider:
        def __init__(self, cfg):
            raise RuntimeError("bad key")

    provider_classes = {"claude": BrokenProvider, "gemini": BrokenProvider, "openai": BrokenProvider}
    providers = build_all_providers(multi_model_config, provider_classes)
    assert providers == {}


# ---------------------------------------------------------------------------
# CouncilRunner.run
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_round():
    return Round(number=1, responses=[])


@pytest.fixture
def fake_result(fake_round):
    q = Question(text="Test question", source="cli")
    return DebateResult(
        question=q,
        rounds=[fake_round],
        synthesis="Final synthesis",
        synthesizer="openai",
        total_duration_sec=1.0,
        panel_mode="custom",
    )


async def test_runner_run_returns_debate_result(all_providers, multi_model_config, tmp_path, fake_round, fake_result):
    request = RunRequest(
        question=Question(text="Test question", source="cli"),
        panel_names=["claude", "gemini"],
        synthesizer_name="openai",
        rounds=1,
        policy=RunPolicy.default(),
        panel_mode="custom",
    )

    with (
        patch("ai_council.orchestrator.run_debate", new=AsyncMock(return_value=DebateOutcome(rounds=[fake_round]))),
        patch("ai_council.orchestrator.synthesize", new=AsyncMock(return_value=fake_result)),
        patch("ai_council.orchestrator.save_to_file", return_value=[tmp_path / "out.md"]),
        patch("ai_council.orchestrator.print_round_summary"),
        patch("ai_council.orchestrator.print_synthesis"),
        patch("ai_council.orchestrator.print_cost_summary"),
    ):
        runner = CouncilRunner(all_providers, multi_model_config)
        result = await runner.run(request, output_dir=tmp_path)

    assert result is fake_result


# ---------------------------------------------------------------------------
# P1-9 — a synthesis failure must not discard the paid-for debate
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_outcome():
    """A completed two-seat debate — everything below has already been paid for."""
    responses = [
        ModelResponse("claude", "claude-opus-4-8", 1, "Claude's round 1 argument.", 1.0, 10,
                      input_tokens=5, output_tokens=5),
        ModelResponse("gemini", "gemini-3-pro", 1, "Gemini's round 1 argument.", 1.1, 12,
                      input_tokens=6, output_tokens=6),
    ]
    return DebateOutcome(
        rounds=[Round(number=1, responses=responses)],
        provider_statuses={"claude": "ok", "gemini": "ok"},
    )


def _synthesis_failure_request():
    return RunRequest(
        question=Question(text="Should we use YAML or JSON?", source="cli"),
        panel_names=["claude", "gemini"],
        synthesizer_name="openai",
        rounds=1,
        policy=RunPolicy.default(),
        panel_mode="custom",
    )


async def _run_with_failing_synthesis(
    all_providers, multi_model_config, tmp_path, outcome, exc
):
    """Drive the REAL writer path with synthesis raising — no save_* is patched out.

    P1-16: both orchestration functions are mocked out of existence in every existing test
    that names them, so the sequencing this fix changes has no coverage at the site that
    implements it. These tests exercise the actual seam.
    """
    with (
        patch("ai_council.orchestrator.run_debate", new=AsyncMock(return_value=outcome)),
        patch("ai_council.orchestrator.synthesize", new=AsyncMock(side_effect=exc)),
        patch("ai_council.orchestrator.print_round_summary"),
        patch("ai_council.orchestrator.print_synthesis"),
        patch("ai_council.orchestrator.print_cost_summary"),
    ):
        runner = CouncilRunner(all_providers, multi_model_config)
        with pytest.raises(type(exc)):
            await runner.run(_synthesis_failure_request(), output_dir=tmp_path)


@pytest.mark.parametrize(
    "exc",
    [
        ProviderError("openai", "API error"),
        RuntimeError("Synthesizer openai returned empty content"),
    ],
    ids=["provider-error", "empty-content"],
)
async def test_synthesis_failure_preserves_the_transcript(
    all_providers, multi_model_config, tmp_path, populated_outcome, exc
):
    """P1-9: synthesize() raised before ANY writer ran (orchestrator.py:153 precedes :204), so
    a synthesizer hiccup — the single most replaceable call in the pipeline — destroyed the
    transcript, the metrics sidecar and every round the run had already paid for."""
    await _run_with_failing_synthesis(
        all_providers, multi_model_config, tmp_path, populated_outcome, exc
    )

    transcripts = list(tmp_path.glob("council-out-*.md"))
    assert len(transcripts) == 1, "the debate transcript must survive a synthesis failure"
    content = transcripts[0].read_text(encoding="utf-8")
    assert "SYNTHESIS FAILED" in content  # stated plainly, never a silent empty synthesis
    assert "Claude's round 1 argument." in content
    assert "Gemini's round 1 argument." in content


async def test_synthesis_failure_emits_no_verdict_package(
    all_providers, multi_model_config, tmp_path, populated_outcome
):
    """There is no verdict without synthesis. The package hardcodes exit_semantics: 0, so
    emitting one here would assert a usable verdict while the process exits 1."""
    await _run_with_failing_synthesis(
        all_providers, multi_model_config, tmp_path, populated_outcome,
        ProviderError("openai", "API error"),
    )
    assert list(tmp_path.glob("council-verdict-*.json")) == []
    assert list(tmp_path.glob("council-minority-*.md")) == []


async def test_synthesis_failure_still_writes_the_metrics_sidecar(
    all_providers, multi_model_config, tmp_path, populated_outcome
):
    """The run was billed for every panelist — the cost record must survive with it."""
    await _run_with_failing_synthesis(
        all_providers, multi_model_config, tmp_path, populated_outcome,
        ProviderError("openai", "API error"),
    )
    sidecars = list(tmp_path.glob("council-out-*_metrics.json"))
    assert len(sidecars) == 1
    data = json.loads(sidecars[0].read_text(encoding="utf-8"))
    booked = {c["provider"] for c in data["calls"]}
    assert {"claude", "gemini"} <= booked  # every paid-for panel call is still on the ledger


async def test_synthesis_failure_marks_the_run_degraded(
    all_providers, multi_model_config, tmp_path, populated_outcome
):
    """The transcript must say WHY it has no synthesis, not merely lack one."""
    await _run_with_failing_synthesis(
        all_providers, multi_model_config, tmp_path, populated_outcome,
        ProviderError("openai", "boom-detail-9987"),
    )
    content = list(tmp_path.glob("council-out-*.md"))[0].read_text(encoding="utf-8")
    assert "DEGRADED" in content
    assert "boom-detail-9987" in content  # the original cause is preserved, not swallowed


async def test_runner_run_raises_when_panel_too_small(two_providers, multi_model_config, tmp_path):
    """Panel with no available providers raises RuntimeError."""
    request = RunRequest(
        question=Question(text="Q", source="cli"),
        panel_names=["grok", "deepseek"],  # neither in two_providers
        synthesizer_name="claude",
        rounds=1,
        policy=RunPolicy.default(),
        panel_mode="custom",
    )
    runner = CouncilRunner(two_providers, multi_model_config)
    with pytest.raises(RuntimeError, match="at least"):
        await runner.run(request, output_dir=tmp_path)


async def test_runner_run_uses_output_dir_from_config_when_none(
    all_providers, multi_model_config, fake_round, fake_result
):
    """When output_dir=None, runner uses config.defaults.output_dir."""
    request = RunRequest(
        question=Question(text="Q", source="cli"),
        panel_names=["claude", "gemini"],
        synthesizer_name="openai",
        rounds=1,
        policy=RunPolicy.default(),
        panel_mode="custom",
    )

    saved_path = multi_model_config.defaults.output_dir / "out.md"

    with (
        patch("ai_council.orchestrator.run_debate", new=AsyncMock(return_value=DebateOutcome(rounds=[fake_round]))),
        patch("ai_council.orchestrator.synthesize", new=AsyncMock(return_value=fake_result)),
        patch("ai_council.orchestrator.save_to_file", return_value=[saved_path]) as mock_save,
        patch("ai_council.orchestrator.print_round_summary"),
        patch("ai_council.orchestrator.print_synthesis"),
        patch("ai_council.orchestrator.print_cost_summary"),
    ):
        runner = CouncilRunner(all_providers, multi_model_config)
        await runner.run(request, output_dir=None)

    _, call_kwargs = mock_save.call_args
    assert mock_save.call_args[0][1] == multi_model_config.defaults.output_dir


async def test_runner_passes_target_paths_to_save_to_file(
    all_providers, multi_model_config, tmp_path, fake_round, fake_result
):
    """RunRequest.target_paths is forwarded to save_to_file."""

    target = tmp_path / "mirror" / "docs" / "decisions" / "transcripts"
    request = RunRequest(
        question=Question(text="Q", source="cli"),
        panel_names=["claude", "gemini"],
        synthesizer_name="openai",
        rounds=1,
        policy=RunPolicy.default(),
        panel_mode="custom",
        target_paths=[target],
    )

    with (
        patch("ai_council.orchestrator.run_debate", new=AsyncMock(return_value=DebateOutcome(rounds=[fake_round]))),
        patch("ai_council.orchestrator.synthesize", new=AsyncMock(return_value=fake_result)),
        patch("ai_council.orchestrator.save_to_file", return_value=[tmp_path / "out.md"]) as mock_save,
        patch("ai_council.orchestrator.print_round_summary"),
        patch("ai_council.orchestrator.print_synthesis"),
        patch("ai_council.orchestrator.print_cost_summary"),
    ):
        runner = CouncilRunner(all_providers, multi_model_config)
        await runner.run(request, output_dir=tmp_path)

    call_kwargs = mock_save.call_args.kwargs
    assert call_kwargs["target_paths"] == [target]


async def test_runner_empty_target_paths_by_default(
    all_providers, multi_model_config, tmp_path, fake_round, fake_result
):
    """When no target_paths on RunRequest, save_to_file receives empty list."""
    request = RunRequest(
        question=Question(text="Q", source="cli"),
        panel_names=["claude", "gemini"],
        synthesizer_name="openai",
        rounds=1,
        policy=RunPolicy.default(),
        panel_mode="custom",
        # target_paths defaults to []
    )

    with (
        patch("ai_council.orchestrator.run_debate", new=AsyncMock(return_value=DebateOutcome(rounds=[fake_round]))),
        patch("ai_council.orchestrator.synthesize", new=AsyncMock(return_value=fake_result)),
        patch("ai_council.orchestrator.save_to_file", return_value=[tmp_path / "out.md"]) as mock_save,
        patch("ai_council.orchestrator.print_round_summary"),
        patch("ai_council.orchestrator.print_synthesis"),
        patch("ai_council.orchestrator.print_cost_summary"),
    ):
        runner = CouncilRunner(all_providers, multi_model_config)
        await runner.run(request, output_dir=tmp_path)

    call_kwargs = mock_save.call_args.kwargs
    assert call_kwargs.get("target_paths", []) == []


# --- #18 crux check flows through the orchestrator ---


async def test_runner_passes_crux_service_into_run_debate(
    all_providers, multi_model_config, fake_round, fake_result, tmp_path
):
    """The service is BUILT in the orchestrator and INJECTED — never constructed in debate.py."""
    from ai_council.models import CruxArtifact, CruxStatus

    artifact = CruxArtifact(status=CruxStatus.GROUNDED, crux_claim="X happens.")
    sentinel = object()
    run_debate_mock = AsyncMock(
        return_value=DebateOutcome(rounds=[fake_round], crux=artifact)
    )
    synthesize_mock = AsyncMock(return_value=fake_result)

    request = RunRequest(
        question=Question(text="Q", source="cli"),
        panel_names=["claude", "gemini"],
        synthesizer_name="openai",
        rounds=2,
        policy=RunPolicy.default(),
        panel_mode="custom",
    )

    with (
        patch("ai_council.orchestrator.run_debate", new=run_debate_mock),
        patch("ai_council.orchestrator.synthesize", new=synthesize_mock),
        patch("ai_council.orchestrator.build_crux_check_service", return_value=sentinel),
        patch("ai_council.orchestrator.save_to_file", return_value=[tmp_path / "out.md"]),
        patch("ai_council.orchestrator.print_round_summary"),
        patch("ai_council.orchestrator.print_synthesis"),
        patch("ai_council.orchestrator.print_cost_summary"),
    ):
        runner = CouncilRunner(all_providers, multi_model_config)
        await runner.run(request, output_dir=tmp_path)

    assert run_debate_mock.await_args.kwargs["crux_check"] is sentinel
    # ...and the artifact comes back out and reaches synthesis.
    assert synthesize_mock.await_args.kwargs["crux"] is artifact


async def test_runner_crux_service_is_none_when_unconfigured(
    all_providers, multi_model_config, fake_round, fake_result, tmp_path
):
    """No crux_check: section → the debate runs exactly as it did pre-#18."""
    multi_model_config.crux_check = None
    run_debate_mock = AsyncMock(return_value=DebateOutcome(rounds=[fake_round]))

    request = RunRequest(
        question=Question(text="Q", source="cli"),
        panel_names=["claude", "gemini"],
        synthesizer_name="openai",
        rounds=2,
        policy=RunPolicy.default(),
        panel_mode="custom",
    )

    with (
        patch("ai_council.orchestrator.run_debate", new=run_debate_mock),
        patch("ai_council.orchestrator.synthesize", new=AsyncMock(return_value=fake_result)),
        patch("ai_council.orchestrator.save_to_file", return_value=[tmp_path / "out.md"]),
        patch("ai_council.orchestrator.print_round_summary"),
        patch("ai_council.orchestrator.print_synthesis"),
        patch("ai_council.orchestrator.print_cost_summary"),
    ):
        runner = CouncilRunner(all_providers, multi_model_config)
        await runner.run(request, output_dir=tmp_path)

    assert run_debate_mock.await_args.kwargs["crux_check"] is None
