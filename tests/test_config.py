"""Tests for config/config_loader.py."""

from pathlib import Path

import pytest
import yaml

from config.config_loader import (
    AppConfig,
    ModelConfig,
    PromptsConfig,
    load_config,
)


@pytest.fixture
def minimal_settings(tmp_path: Path) -> Path:
    """Write a minimal valid settings.yaml to a temp path."""
    settings = {
        "defaults": {
            "rounds": 2,
            "max_rounds": 3,
            "output_dir": "./output",
            "synthesizer": "claude",
            "default_panel": ["claude"],
            "full_panel": ["claude", "openai"],
        },
        "models": {
            "claude": {
                "sdk": "anthropic",
                "model": "claude-opus-4-6",
                "api_key_env": "TEST_CLAUDE_KEY",
                "timeout_sec": 120,
                "max_tokens": 8192,
            }
        },
        "prompts": {
            "initial": "{persona}\nAnswer: {question}",
            "critique": "{persona}\nRound {round}. Q: {question}\n{previous_responses_anonymized}",
            "synthesis": "Q: {question}\n{full_transcript}",
        },
        "personas": {
            "claude": "You are a Systems Architect.",
            "openai": "You are a Product Architect.",
        },
    }
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    return path


def test_load_config_returns_app_config(minimal_settings):
    config = load_config(minimal_settings)
    assert isinstance(config, AppConfig)


def test_load_config_defaults(minimal_settings):
    config = load_config(minimal_settings)
    assert config.defaults.rounds == 2
    assert config.defaults.max_rounds == 3
    assert config.defaults.synthesizer == "claude"
    assert isinstance(config.defaults.output_dir, Path)


def test_load_config_default_panel(minimal_settings):
    config = load_config(minimal_settings)
    assert config.defaults.default_panel == ["claude"]
    assert config.defaults.full_panel == ["claude", "openai"]


def test_load_config_models(minimal_settings):
    config = load_config(minimal_settings)
    assert "claude" in config.models
    assert isinstance(config.models["claude"], ModelConfig)
    assert config.models["claude"].model == "claude-opus-4-6"


def test_load_config_prompts(minimal_settings):
    config = load_config(minimal_settings)
    assert isinstance(config.prompts, PromptsConfig)
    assert "{question}" in config.prompts.initial


def test_load_config_personas(minimal_settings):
    config = load_config(minimal_settings)
    assert isinstance(config.prompts.personas, dict)
    assert "claude" in config.prompts.personas
    assert "Systems Architect" in config.prompts.personas["claude"]
    assert "openai" in config.prompts.personas


def test_load_config_available_providers_with_key(minimal_settings, monkeypatch):
    monkeypatch.setenv("TEST_CLAUDE_KEY", "sk-test-key")
    config = load_config(minimal_settings)
    assert "claude" in config.available_providers


def test_load_config_no_available_providers_without_key(minimal_settings, monkeypatch):
    monkeypatch.delenv("TEST_CLAUDE_KEY", raising=False)
    config = load_config(minimal_settings)
    assert "claude" not in config.available_providers


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/settings.yaml"))


def test_model_config_base_url_optional(minimal_settings):
    config = load_config(minimal_settings)
    assert config.models["claude"].base_url is None


def test_load_config_personas_empty_when_missing(tmp_path: Path):
    """Personas section is optional — should default to empty dict."""
    settings = {
        "defaults": {
            "rounds": 1,
            "max_rounds": 2,
            "output_dir": "./output",
            "synthesizer": "claude",
            "default_panel": ["claude"],
            "full_panel": ["claude"],
        },
        "models": {
            "claude": {
                "sdk": "anthropic",
                "model": "claude-opus-4-6",
                "api_key_env": "TEST_KEY",
                "timeout_sec": 60,
                "max_tokens": 4096,
            }
        },
        "prompts": {
            "initial": "Q: {question}",
            "critique": "Q: {question}",
            "synthesis": "Q: {question}\n{full_transcript}",
        },
        # No "personas" key
    }
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    config = load_config(path)
    assert config.prompts.personas == {}


# ---------------------------------------------------------------------------
# target_projects
# ---------------------------------------------------------------------------


def _minimal_base(tmp_path: Path) -> dict:
    return {
        "defaults": {
            "rounds": 1,
            "max_rounds": 2,
            "output_dir": "./output",
            "synthesizer": "claude",
            "default_panel": ["claude"],
            "full_panel": ["claude"],
        },
        "models": {
            "claude": {
                "sdk": "anthropic",
                "model": "claude-opus-4-6",
                "api_key_env": "TEST_KEY",
                "timeout_sec": 60,
                "max_tokens": 4096,
            }
        },
        "prompts": {
            "initial": "Q: {question}",
            "critique": "Q: {question}",
            "synthesis": "Q: {question}\n{full_transcript}",
        },
    }


def test_target_projects_valid_single_entry(tmp_path: Path):
    settings = _minimal_base(tmp_path)
    settings["target_projects"] = {".dev-knowledge": "C:/Dev/.dev-knowledge"}
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    config = load_config(path)
    assert ".dev-knowledge" in config.target_projects
    # Compare as Path objects to handle forward/backslash normalization on Windows
    assert Path(config.target_projects[".dev-knowledge"]) == Path("C:/Dev/.dev-knowledge")


def test_target_projects_valid_empty_map(tmp_path: Path):
    settings = _minimal_base(tmp_path)
    settings["target_projects"] = {}
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    config = load_config(path)
    assert config.target_projects == {}


def test_target_projects_absent_defaults_to_empty(tmp_path: Path):
    settings = _minimal_base(tmp_path)
    # No target_projects key at all
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    config = load_config(path)
    assert config.target_projects == {}


def test_target_projects_list_raises(tmp_path: Path):
    settings = _minimal_base(tmp_path)
    settings["target_projects"] = [".dev-knowledge"]
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="target_projects must be a mapping"):
        load_config(path)


def test_target_projects_non_string_value_raises(tmp_path: Path):
    settings = _minimal_base(tmp_path)
    settings["target_projects"] = {".dev-knowledge": 42}
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="string->string"):
        load_config(path)


def test_target_projects_paths_are_expanded(tmp_path: Path):
    """Paths with ~ are expanded at load time, not stored literally."""
    from pathlib import Path as P
    settings = _minimal_base(tmp_path)
    settings["target_projects"] = {".dev-knowledge": "~/Dev/.dev-knowledge"}
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    config = load_config(path)
    stored = config.target_projects[".dev-knowledge"]
    assert "~" not in stored
    assert str(P.home()) in stored
