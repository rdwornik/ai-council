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
    settings["dev_root"] = str(tmp_path)
    settings["target_projects"] = [".dev-knowledge"]
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    config = load_config(path)
    assert ".dev-knowledge" in config.target_projects
    assert config.dev_root == tmp_path


def test_target_projects_valid_empty_list(tmp_path: Path):
    settings = _minimal_base(tmp_path)
    settings["target_projects"] = []
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    config = load_config(path)
    assert config.target_projects == []


def test_target_projects_absent_defaults_to_empty(tmp_path: Path):
    settings = _minimal_base(tmp_path)
    # No target_projects key at all
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    config = load_config(path)
    assert config.target_projects == []


def test_target_projects_dict_raises_migration_error(tmp_path: Path):
    """Old dict schema must fail loud with migration hint."""
    settings = _minimal_base(tmp_path)
    settings["target_projects"] = {".dev-knowledge": "C:/Dev/.dev-knowledge"}
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="ADR-43 amendment cycle 1"):
        load_config(path)


def test_target_projects_dict_error_mentions_old_and_new_shape(tmp_path: Path):
    settings = _minimal_base(tmp_path)
    settings["target_projects"] = {".dev-knowledge": "C:/Dev/.dev-knowledge"}
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        load_config(path)
    msg = str(exc_info.value)
    assert "list" in msg
    assert "dict" in msg


def test_target_projects_non_string_item_raises(tmp_path: Path):
    settings = _minimal_base(tmp_path)
    settings["dev_root"] = str(tmp_path)
    settings["target_projects"] = [42]
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="items must be strings"):
        load_config(path)


def test_target_projects_duplicate_names_raises(tmp_path: Path):
    settings = _minimal_base(tmp_path)
    settings["dev_root"] = str(tmp_path)
    settings["target_projects"] = [".dev-knowledge", ".dev-knowledge"]
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate project name"):
        load_config(path)


def test_dev_root_required_when_target_projects_nonempty(tmp_path: Path):
    settings = _minimal_base(tmp_path)
    settings["target_projects"] = [".dev-knowledge"]
    # No dev_root key
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="dev_root is required"):
        load_config(path)


def test_dev_root_must_be_existing_directory(tmp_path: Path):
    settings = _minimal_base(tmp_path)
    settings["dev_root"] = str(tmp_path / "nonexistent_dir")
    settings["target_projects"] = [".dev-knowledge"]
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="dev_root must point to existing directory"):
        load_config(path)


def test_dev_root_expanduser_normalized(tmp_path: Path):
    """dev_root with ~ is expanded at load time."""
    from pathlib import Path as P
    settings = _minimal_base(tmp_path)
    # Use a path that definitely exists after expanduser — home dir itself
    settings["dev_root"] = "~"
    settings["target_projects"] = [".dev-knowledge"]
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.dump(settings), encoding="utf-8")
    config = load_config(path)
    assert config.dev_root == P.home()
    assert "~" not in str(config.dev_root)
