"""Tests for CouncilRunner and module-level runner helpers."""

from unittest.mock import AsyncMock, patch

import pytest

from ai_council.models import DebateOutcome, DebateResult, Question, Round, RunRequest
from ai_council.policy import RunPolicy
from ai_council.runner import (
    CouncilRunner,
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


async def test_runner_run_uses_output_dir_from_config_when_none(all_providers, multi_model_config, fake_round, fake_result):
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
