"""Tests for CLI panel/synthesizer selection logic in src/cli.py."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from ai_council.cli import main
from ai_council.models import DebateResult, Question, Round
from ai_council.runner import (
    determine_panel as _determine_panel,
)
from ai_council.runner import (
    exclude_synthesizer_from_panel as _exclude_synthesizer_from_panel,
)
from ai_council.runner import (
    pick_synthesizer as _pick_non_participant_synthesizer,
)
from config.config_loader import AppConfig, DefaultsConfig, ModelConfig, PromptsConfig
from tests.conftest import MockProvider

# ---------------------------------------------------------------------------
# Helpers for CLI routing tests
# ---------------------------------------------------------------------------


def _make_test_config(
    tmp_path: Path,
    dev_root: Path | None = None,
    target_projects: list[str] | None = None,
) -> AppConfig:
    """Minimal AppConfig with controllable routing fields for routing tests."""
    model = ModelConfig(
        name="claude", sdk="anthropic", model="claude-test",
        api_key_env="TEST_KEY", timeout_sec=60, max_tokens=1024,
    )
    defaults = DefaultsConfig(
        rounds=1, max_rounds=2,
        output_dir=tmp_path / "output",
        synthesizer="claude",
        default_panel=["claude"],
        full_panel=["claude"],
    )
    prompts = PromptsConfig(
        initial="{persona}\n{question}",
        critique="{persona}\nRound {round}. {question}\n{previous_responses_anonymized}",
        synthesis="Q: {question}\n{full_transcript}",
        personas={"claude": "Be an architect."},
    )
    return AppConfig(
        defaults=defaults,
        models={"claude": model},
        prompts=prompts,
        available_providers={"claude"},
        dev_root=dev_root,
        target_projects=target_projects or [],
    )


@pytest.fixture
def mock_all_providers():
    return {
        "claude": MockProvider("claude"),
        "gemini": MockProvider("gemini"),
        "deepseek": MockProvider("deepseek"),
        "openai": MockProvider("openai"),
        "grok": MockProvider("grok"),
    }


def test_determine_panel_default(sample_defaults_config, sample_app_config):
    panel, mode = _determine_panel(sample_app_config, models_arg=None, full_flag=False)
    assert panel == sample_app_config.defaults.default_panel
    assert mode == "default"


def test_determine_panel_full(sample_defaults_config, sample_app_config):
    panel, mode = _determine_panel(sample_app_config, models_arg=None, full_flag=True)
    assert panel == sample_app_config.defaults.full_panel
    assert mode == "full"


def test_determine_panel_custom_models_arg(sample_app_config):
    panel, mode = _determine_panel(
        sample_app_config, models_arg="claude,openai", full_flag=False
    )
    assert panel == ["claude", "openai"]
    assert mode == "custom"


def test_determine_panel_models_arg_overrides_full(sample_app_config):
    """--models should override --full."""
    panel, mode = _determine_panel(
        sample_app_config, models_arg="claude,grok", full_flag=True
    )
    assert panel == ["claude", "grok"]
    assert mode == "custom"


def test_cli_full_flag_uses_full_panel(sample_app_config):
    panel, mode = _determine_panel(sample_app_config, models_arg=None, full_flag=True)
    assert panel == sample_app_config.defaults.full_panel
    assert mode == "full"


def test_determine_panel_lite_uses_default_panel(sample_app_config):
    """--lite passes full_flag=False, yielding the 3-model default panel."""
    panel, mode = _determine_panel(sample_app_config, models_arg=None, full_flag=False)
    assert panel == sample_app_config.defaults.default_panel
    assert mode == "default"


def test_determine_panel_full_flag_true_still_works(sample_app_config):
    """--full (now no-op, still passes full_flag=True) still returns full panel."""
    panel, mode = _determine_panel(sample_app_config, models_arg=None, full_flag=True)
    assert panel == sample_app_config.defaults.full_panel
    assert mode == "full"


def test_non_participant_synthesizer_not_in_panel(mock_all_providers):
    panel_names = ["claude", "gemini", "deepseek"]
    synth, is_participant = _pick_non_participant_synthesizer(
        mock_all_providers, panel_names, preferred="openai"
    )
    assert synth.name() not in panel_names
    assert is_participant is False


def test_non_participant_preferred_chosen(mock_all_providers):
    panel_names = ["claude", "gemini", "deepseek"]
    synth, is_participant = _pick_non_participant_synthesizer(
        mock_all_providers, panel_names, preferred="openai"
    )
    assert synth.name() == "openai"
    assert is_participant is False


def test_non_participant_falls_back_when_preferred_not_available(mock_all_providers):
    """If preferred synthesizer is in panel, pick another non-participant."""
    panel_names = ["claude", "gemini", "deepseek"]
    synth, is_participant = _pick_non_participant_synthesizer(
        mock_all_providers,
        panel_names,
        preferred="claude",  # claude is in panel
    )
    assert synth.name() not in panel_names
    assert is_participant is False


def test_non_participant_falls_back_when_all_in_panel():
    """When all available providers are in the panel, is_participant=True."""
    all_providers = {
        "claude": MockProvider("claude"),
        "gemini": MockProvider("gemini"),
    }
    panel_names = ["claude", "gemini"]
    synth, is_participant = _pick_non_participant_synthesizer(
        all_providers, panel_names, preferred="claude"
    )
    assert is_participant is True
    assert synth.name() == "claude"


def test_non_participant_all_in_panel_no_preferred():
    """Fallback to first available when preferred not available and all in panel."""
    all_providers = {
        "claude": MockProvider("claude"),
        "gemini": MockProvider("gemini"),
    }
    panel_names = ["claude", "gemini"]
    synth, is_participant = _pick_non_participant_synthesizer(
        all_providers,
        panel_names,
        preferred="openai",  # not available
    )
    assert is_participant is True
    assert synth.name() in {"claude", "gemini"}


# ---------------------------------------------------------------------------
# Mode-scoped health-check gating (Issue 2)
# ---------------------------------------------------------------------------


def _modes_with_research() -> dict:
    """Minimal modes dict including 'research' with its 'r' alias — enough for resolve_mode."""
    from config.config_loader import ModeConfig

    def _m(aliases, default=False, max_rounds=2):
        return ModeConfig(
            description="", emoji="", aliases=aliases, default=default,
            max_rounds=max_rounds, token_budget=4096,
        )

    return {
        "pick": _m(["p", "decide", "d"], default=True),
        "ideas": _m(["i"], max_rounds=1),
        "judge": _m(["j"]),
        "research": _m(["r"], max_rounds=1),
    }


def _research_cfg(summary_model: str = "deepseek"):
    from config.config_loader import ResearchConfig

    return ResearchConfig(
        default_providers=["perplexity"],
        deep_providers=["perplexity"],
        cache_dir=Path("/tmp/cache"),
        cache_ttl_days=7,
        summary_max_tokens=2500,
        summary_model=summary_model,
    )


def test_health_check_targets_research_returns_summarizer_only_nonblocking():
    """--mode research → only the summarizer is health-checked, and the gate is non-blocking."""
    from ai_council.cli import _select_health_check_targets

    providers = {
        "claude": MockProvider("claude"),
        "claude-sonnet": MockProvider("claude-sonnet"),
        "gemini": MockProvider("gemini"),
        "deepseek": MockProvider("deepseek"),
        "openai": MockProvider("openai"),
        "grok": MockProvider("grok"),
    }
    targets, blocking = _select_health_check_targets(
        providers,
        cli_mode_arg="r",
        modes=_modes_with_research(),
        research_cfg=_research_cfg("deepseek"),
    )
    assert list(targets.keys()) == ["deepseek"]
    assert blocking is False


def test_health_check_targets_research_canonical_name():
    """'research' (canonical) resolves identically to 'r' (alias)."""
    from ai_council.cli import _select_health_check_targets

    providers = {"deepseek": MockProvider("deepseek"), "claude": MockProvider("claude")}
    targets, blocking = _select_health_check_targets(
        providers,
        cli_mode_arg="research",
        modes=_modes_with_research(),
        research_cfg=_research_cfg("deepseek"),
    )
    assert set(targets.keys()) == {"deepseek"}
    assert blocking is False


def test_health_check_targets_research_summarizer_missing_returns_empty_nonblocking():
    """If the summarizer model isn't in the provider pool, return empty + non-blocking — never gate."""
    from ai_council.cli import _select_health_check_targets

    providers = {"claude": MockProvider("claude"), "gemini": MockProvider("gemini")}
    targets, blocking = _select_health_check_targets(
        providers,
        cli_mode_arg="r",
        modes=_modes_with_research(),
        research_cfg=_research_cfg("deepseek"),  # deepseek not in providers
    )
    assert targets == {}
    assert blocking is False


def test_health_check_targets_debate_mode_returns_full_pool_blocking():
    """--mode pick / ideas / judge → full provider pool, blocking gate (current behaviour)."""
    from ai_council.cli import _select_health_check_targets

    providers = {
        "claude": MockProvider("claude"),
        "gemini": MockProvider("gemini"),
        "deepseek": MockProvider("deepseek"),
    }
    for mode_arg in ("pick", "p", "ideas", "i", "judge", "j"):
        targets, blocking = _select_health_check_targets(
            providers,
            cli_mode_arg=mode_arg,
            modes=_modes_with_research(),
            research_cfg=_research_cfg("deepseek"),
        )
        assert set(targets.keys()) == {"claude", "gemini", "deepseek"}, mode_arg
        assert blocking is True, mode_arg


def test_health_check_targets_no_mode_arg_defaults_to_full_blocking():
    """No --mode flag (auto-detect path) → full pool, blocking. Pre-existing behaviour."""
    from ai_council.cli import _select_health_check_targets

    providers = {"claude": MockProvider("claude"), "gemini": MockProvider("gemini")}
    targets, blocking = _select_health_check_targets(
        providers,
        cli_mode_arg=None,
        modes=_modes_with_research(),
        research_cfg=_research_cfg("deepseek"),
    )
    assert set(targets.keys()) == {"claude", "gemini"}
    assert blocking is True


def test_health_check_targets_research_without_research_cfg_falls_back_to_full():
    """If config.research is None (mis-config), don't crash — fall back to full pool / blocking."""
    from ai_council.cli import _select_health_check_targets

    providers = {"claude": MockProvider("claude")}
    targets, blocking = _select_health_check_targets(
        providers,
        cli_mode_arg="r",
        modes=_modes_with_research(),
        research_cfg=None,
    )
    assert set(targets.keys()) == {"claude"}
    assert blocking is True


# --- _exclude_synthesizer_from_panel tests ---


def test_exclude_synthesizer_from_full_panel():
    """Full panel including synthesizer: synthesizer removed, leaving 4 debaters."""
    all_providers = {
        n: MockProvider(n) for n in ["claude", "gemini", "deepseek", "openai", "grok"]
    }
    panel = ["claude", "gemini", "deepseek", "openai", "grok"]
    result = _exclude_synthesizer_from_panel(panel, "openai", all_providers)
    assert "openai" not in result
    assert len(result) == 4


def test_exclude_synthesizer_not_in_panel():
    """When synthesizer is already absent from panel, no change."""
    all_providers = {n: MockProvider(n) for n in ["claude", "gemini", "deepseek"]}
    panel = ["claude", "gemini", "deepseek"]
    result = _exclude_synthesizer_from_panel(panel, "openai", all_providers)
    assert result == panel


def test_exclude_synthesizer_keeps_when_only_two_left():
    """When removing synthesizer would leave fewer than 2 available debaters, keep it."""
    all_providers = {"claude": MockProvider("claude"), "openai": MockProvider("openai")}
    panel = ["claude", "openai"]
    result = _exclude_synthesizer_from_panel(panel, "openai", all_providers)
    assert result == panel  # can't remove, would leave only 1 debater


# ---------------------------------------------------------------------------
# --target-project CLI flag routing
# ---------------------------------------------------------------------------


def test_cli_unknown_target_project_exits_nonzero(tmp_path: Path) -> None:
    """Unknown --target-project name exits non-zero before health check runs."""
    config = _make_test_config(tmp_path, dev_root=tmp_path, target_projects=[".dev-knowledge"])

    with patch("ai_council.cli.load_config", return_value=config):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--skip-health-check", "--target-project", "no-such-project", "test question"],
        )

    assert result.exit_code != 0
    assert "Unknown target-project" in result.output
    assert "no-such-project" in result.output


def test_cli_unknown_target_project_lists_known_names(tmp_path: Path) -> None:
    """Error message for unknown target-project lists known target names."""
    config = _make_test_config(tmp_path, dev_root=tmp_path, target_projects=[".dev-knowledge"])

    with patch("ai_council.cli.load_config", return_value=config):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--skip-health-check", "--target-project", "ghost", "test question"],
        )

    assert ".dev-knowledge" in result.output


def test_cli_known_target_project_populates_request(tmp_path: Path) -> None:
    """--target-project with known name populates target_paths on RunRequest."""
    config = _make_test_config(tmp_path, dev_root=tmp_path, target_projects=[".dev-knowledge"])
    fake_provider = MockProvider("claude")
    fake_round = Round(number=1, responses=[])
    fake_result = DebateResult(
        question=Question(text="test", source="cli"),
        rounds=[fake_round],
        synthesis="Result",
        synthesizer="claude",
        total_duration_sec=1.0,
        panel_mode="custom",
    )

    captured_request: list = []

    async def _fake_run(request, output_dir=None, output_format="text"):
        captured_request.append(request)
        return fake_result

    with (
        patch("ai_council.cli.load_config", return_value=config),
        patch("ai_council.cli.build_all_providers", return_value={"claude": fake_provider}),
        patch("ai_council.cli.CouncilRunner") as MockRunner,
    ):
        MockRunner.return_value.run = _fake_run
        runner = CliRunner()
        runner.invoke(
            main,
            ["--skip-health-check", "--target-project", ".dev-knowledge", "--mode", "pick", "test question"],
        )

    assert len(captured_request) == 1
    req = captured_request[0]
    assert len(req.target_paths) == 1
    assert ".dev-knowledge" in str(req.target_paths[0])


def test_cli_no_target_project_empty_target_paths(tmp_path: Path) -> None:
    """Without --target-project, target_paths is empty on RunRequest."""
    config = _make_test_config(tmp_path, dev_root=tmp_path, target_projects=[".dev-knowledge"])
    fake_provider = MockProvider("claude")
    fake_round = Round(number=1, responses=[])
    fake_result = DebateResult(
        question=Question(text="test", source="cli"),
        rounds=[fake_round],
        synthesis="Result",
        synthesizer="claude",
        total_duration_sec=1.0,
        panel_mode="custom",
    )

    captured_request: list = []

    async def _fake_run(request, output_dir=None, output_format="text"):
        captured_request.append(request)
        return fake_result

    with (
        patch("ai_council.cli.load_config", return_value=config),
        patch("ai_council.cli.build_all_providers", return_value={"claude": fake_provider}),
        patch("ai_council.cli.CouncilRunner") as MockRunner,
    ):
        MockRunner.return_value.run = _fake_run
        runner = CliRunner()
        runner.invoke(
            main,
            ["--skip-health-check", "--mode", "pick", "test question"],
        )

    assert len(captured_request) == 1
    assert captured_request[0].target_paths == []


def test_cli_multiple_target_projects(tmp_path: Path) -> None:
    """Repeated --target-project resolves multiple targets with correct paths."""
    config = _make_test_config(tmp_path, dev_root=tmp_path, target_projects=[".dev-knowledge", "foo"])
    fake_provider = MockProvider("claude")
    fake_round = Round(number=1, responses=[])
    fake_result = DebateResult(
        question=Question(text="test", source="cli"),
        rounds=[fake_round],
        synthesis="Result",
        synthesizer="claude",
        total_duration_sec=1.0,
        panel_mode="custom",
    )

    captured_request: list = []

    async def _fake_run(request, output_dir=None, output_format="text"):
        captured_request.append(request)
        return fake_result

    with (
        patch("ai_council.cli.load_config", return_value=config),
        patch("ai_council.cli.build_all_providers", return_value={"claude": fake_provider}),
        patch("ai_council.cli.CouncilRunner") as MockRunner,
    ):
        MockRunner.return_value.run = _fake_run
        runner = CliRunner()
        runner.invoke(
            main,
            [
                "--skip-health-check", "--mode", "pick",
                "--target-project", ".dev-knowledge",
                "--target-project", "foo",
                "test question",
            ],
        )

    _transcripts = Path("docs") / "decisions" / "transcripts"
    assert len(captured_request) == 1
    paths = captured_request[0].target_paths
    assert len(paths) == 2
    assert paths[0] == tmp_path / ".dev-knowledge" / _transcripts
    assert paths[1] == tmp_path / "foo" / _transcripts
