"""Tests for CLI panel/synthesizer selection logic in src/cli.py."""

import os
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


# ---------------------------------------------------------------------------
# DOC-3 (#30): empty API-key env var is treated as absent, LOUDLY
# ---------------------------------------------------------------------------


def test_strip_empty_api_keys_treats_empty_as_absent(tmp_path, monkeypatch):
    """DOC-3 (#30): an expected API-key env var that is set-but-EMPTY is stripped from the
    environment (treated as ABSENT) and reported, so it never silently shadows a real .env value
    under load_dotenv(override=False). Guards the cli.py:386 hazard."""
    from ai_council.cli import _strip_empty_api_keys

    config = _make_test_config(tmp_path)  # models={"claude": ModelConfig(api_key_env="TEST_KEY")}
    monkeypatch.setenv("TEST_KEY", "")  # present but empty — the hazard
    stripped = _strip_empty_api_keys(config)
    assert stripped == ["TEST_KEY"]
    assert "TEST_KEY" not in os.environ


def test_strip_empty_api_keys_keeps_present_and_ignores_absent(tmp_path, monkeypatch):
    """A non-empty key is left untouched; an absent key is not reported (no false positives)."""
    from ai_council.cli import _strip_empty_api_keys

    config = _make_test_config(tmp_path)
    monkeypatch.setenv("TEST_KEY", "real-value")
    assert _strip_empty_api_keys(config) == []
    assert os.environ["TEST_KEY"] == "real-value"
    monkeypatch.delenv("TEST_KEY", raising=False)
    assert _strip_empty_api_keys(config) == []


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
    targets, blocking, missing = _select_health_check_targets(
        providers,
        cli_mode_arg="r",
        modes=_modes_with_research(),
        research_cfg=_research_cfg("deepseek"),
    )
    assert list(targets.keys()) == ["deepseek"]
    assert blocking is False
    assert missing is None


def test_health_check_targets_research_canonical_name():
    """'research' (canonical) resolves identically to 'r' (alias)."""
    from ai_council.cli import _select_health_check_targets

    providers = {"deepseek": MockProvider("deepseek"), "claude": MockProvider("claude")}
    targets, blocking, missing = _select_health_check_targets(
        providers,
        cli_mode_arg="research",
        modes=_modes_with_research(),
        research_cfg=_research_cfg("deepseek"),
    )
    assert set(targets.keys()) == {"deepseek"}
    assert blocking is False
    assert missing is None


def test_health_check_targets_research_summarizer_missing_returns_name_for_warning():
    """If the summarizer model failed to build, return empty + non-blocking + the name
    so the caller can emit an explicit WARN instead of silently doing nothing."""
    from ai_council.cli import _select_health_check_targets

    providers = {"claude": MockProvider("claude"), "gemini": MockProvider("gemini")}
    targets, blocking, missing = _select_health_check_targets(
        providers,
        cli_mode_arg="r",
        modes=_modes_with_research(),
        research_cfg=_research_cfg("deepseek"),  # deepseek not in providers
    )
    assert targets == {}
    assert blocking is False
    assert missing == "deepseek"


def test_health_check_targets_debate_mode_returns_full_pool_blocking():
    """--mode pick / ideas / judge → full provider pool, blocking gate (current behaviour)."""
    from ai_council.cli import _select_health_check_targets

    providers = {
        "claude": MockProvider("claude"),
        "gemini": MockProvider("gemini"),
        "deepseek": MockProvider("deepseek"),
    }
    for mode_arg in ("pick", "p", "ideas", "i", "judge", "j"):
        targets, blocking, missing = _select_health_check_targets(
            providers,
            cli_mode_arg=mode_arg,
            modes=_modes_with_research(),
            research_cfg=_research_cfg("deepseek"),
        )
        assert set(targets.keys()) == {"claude", "gemini", "deepseek"}, mode_arg
        assert blocking is True, mode_arg
        assert missing is None, mode_arg


def test_health_check_targets_no_mode_arg_defaults_to_full_blocking():
    """No --mode flag (auto-detect path) → full pool, blocking. Pre-existing behaviour."""
    from ai_council.cli import _select_health_check_targets

    providers = {"claude": MockProvider("claude"), "gemini": MockProvider("gemini")}
    targets, blocking, missing = _select_health_check_targets(
        providers,
        cli_mode_arg=None,
        modes=_modes_with_research(),
        research_cfg=_research_cfg("deepseek"),
    )
    assert set(targets.keys()) == {"claude", "gemini"}
    assert blocking is True
    assert missing is None


def test_health_check_targets_research_without_research_cfg_falls_back_to_full():
    """If config.research is None (mis-config), don't crash — fall back to full pool / blocking."""
    from ai_council.cli import _select_health_check_targets

    providers = {"claude": MockProvider("claude")}
    targets, blocking, missing = _select_health_check_targets(
        providers,
        cli_mode_arg="r",
        modes=_modes_with_research(),
        research_cfg=None,
    )
    assert set(targets.keys()) == {"claude"}
    assert blocking is True
    assert missing is None


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


# ---------------------------------------------------------------------------
# A2: @click.group invocation surface (DefaultGroup preserves bare `council "q"`)
# ---------------------------------------------------------------------------


def test_group_exposes_run_and_doctor() -> None:
    """main is a group with run + doctor subcommands (A2)."""
    import click

    assert isinstance(main, click.Group)
    assert set(main.commands) == {"run", "doctor"}


def test_bare_question_routes_to_run(tmp_path: Path) -> None:
    """`council "q"` (no `run` token) still reaches the run path unchanged."""
    config = _make_test_config(tmp_path)
    with patch("ai_council.cli.load_config", return_value=config):
        result = CliRunner().invoke(main, ["--skip-health-check", "bare question"])
    # Reaches run and proceeds past arg-parsing (no click usage error / exit 2).
    assert result.exit_code != 2


def test_explicit_run_subcommand(tmp_path: Path) -> None:
    """`council run "q"` is equivalent to the bare form."""
    config = _make_test_config(tmp_path)
    with patch("ai_council.cli.load_config", return_value=config):
        result = CliRunner().invoke(main, ["run", "--skip-health-check", "explicit question"])
    assert result.exit_code != 2


def test_modes_flag_at_group_root() -> None:
    """`council --modes` prints modes and exits without needing a subcommand."""
    with patch("ai_council.cli.load_config", side_effect=Exception("no config")):
        result = CliRunner().invoke(main, ["--modes"])
    # eager --modes callback handled it (prints and exits); no crash routing to run.
    assert result.exit_code == 0


def test_group_help_lists_subcommands() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "doctor" in result.output


# ---------------------------------------------------------------------------
# #22: --file frontmatter parsing + precedence (flag > frontmatter > config default)
# ---------------------------------------------------------------------------


def _capture_run_request(config: AppConfig, args: list[str]) -> list:
    """Invoke the CLI and capture the RunRequest passed to CouncilRunner.run."""
    fake_provider = MockProvider("claude")
    fake_result = DebateResult(
        question=Question(text="test", source="cli"),
        rounds=[Round(number=1, responses=[])],
        synthesis="Result",
        synthesizer="claude",
        total_duration_sec=1.0,
        panel_mode="custom",
    )
    captured: list = []

    async def _fake_run(request, output_dir=None, output_format="text"):
        captured.append(request)
        return fake_result

    with (
        patch("ai_council.cli.load_config", return_value=config),
        patch("ai_council.cli.build_all_providers", return_value={"claude": fake_provider}),
        patch("ai_council.cli.CouncilRunner") as MockRunner,
    ):
        MockRunner.return_value.run = _fake_run
        CliRunner().invoke(main, args)
    return captured


def _write_brief(tmp_path: Path, body: str, frontmatter: str = "") -> Path:
    brief = tmp_path / "brief.md"
    content = f"---\n{frontmatter}\n---\n{body}\n" if frontmatter else f"{body}\n"
    brief.write_text(content, encoding="utf-8")
    return brief


def test_file_frontmatter_stripped_no_leak(tmp_path: Path) -> None:
    """#22: YAML frontmatter must not leak into the question text sent to panelists."""
    config = _make_test_config(tmp_path)
    brief = _write_brief(
        tmp_path,
        body="What is the best cache strategy?",
        frontmatter="rounds: 3\nsynthesizer: claude",
    )
    captured = _capture_run_request(config, ["--skip-health-check", "--file", str(brief)])
    assert len(captured) == 1
    text = captured[0].question.text
    assert text == "What is the best cache strategy?"
    assert "rounds:" not in text
    assert "synthesizer:" not in text
    assert "---" not in text


def test_file_rounds_config_default(tmp_path: Path) -> None:
    """#22 tier 3: no flag, no frontmatter -> config default rounds."""
    config = _make_test_config(tmp_path)  # defaults.rounds == 1
    brief = _write_brief(tmp_path, body="A question with no rounds override.")
    captured = _capture_run_request(config, ["--skip-health-check", "--file", str(brief)])
    assert captured[0].rounds == 1


def test_file_rounds_frontmatter_over_default(tmp_path: Path) -> None:
    """#22 tier 2: frontmatter rounds override the config default."""
    config = _make_test_config(tmp_path)
    brief = _write_brief(tmp_path, body="A question.", frontmatter="rounds: 3")
    captured = _capture_run_request(config, ["--skip-health-check", "--file", str(brief)])
    assert captured[0].rounds == 3


def test_file_rounds_flag_wins_over_frontmatter(tmp_path: Path) -> None:
    """#22 tier 1: the CLI --rounds flag wins over frontmatter."""
    config = _make_test_config(tmp_path)
    brief = _write_brief(tmp_path, body="A question.", frontmatter="rounds: 3")
    captured = _capture_run_request(
        config, ["--skip-health-check", "--rounds", "5", "--file", str(brief)]
    )
    assert captured[0].rounds == 5


def test_file_synthesizer_frontmatter_honored(tmp_path: Path) -> None:
    """#22: frontmatter synthesizer is honored and marks the request synthesizer-specified."""
    config = _make_test_config(tmp_path)
    brief = _write_brief(tmp_path, body="A question.", frontmatter="synthesizer: openai")
    captured = _capture_run_request(config, ["--skip-health-check", "--file", str(brief)])
    assert captured[0].synthesizer_name == "openai"
    assert captured[0].synthesizer_specified is True


def test_file_synthesizer_flag_wins_over_frontmatter(tmp_path: Path) -> None:
    """#22 tier 1: the CLI --synthesizer flag wins over frontmatter."""
    config = _make_test_config(tmp_path)
    brief = _write_brief(tmp_path, body="A question.", frontmatter="synthesizer: openai")
    captured = _capture_run_request(
        config, ["--skip-health-check", "--synthesizer", "gemini", "--file", str(brief)]
    )
    assert captured[0].synthesizer_name == "gemini"


def test_file_target_project_frontmatter_resolved(tmp_path: Path) -> None:
    """#22: frontmatter target-project is resolved to target_paths."""
    config = _make_test_config(tmp_path, dev_root=tmp_path, target_projects=[".dev-knowledge"])
    brief = _write_brief(tmp_path, body="A question.", frontmatter="target-project: .dev-knowledge")
    captured = _capture_run_request(config, ["--skip-health-check", "--file", str(brief)])
    assert len(captured[0].target_paths) == 1
    assert ".dev-knowledge" in str(captured[0].target_paths[0])
