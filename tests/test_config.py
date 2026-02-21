"""Tests for config/config_loader.py."""

import os
from pathlib import Path

import pytest
import yaml

from config.config_loader import AppConfig, DefaultsConfig, ModelConfig, PromptsConfig, load_config


@pytest.fixture
def minimal_settings(tmp_path: Path) -> Path:
    """Write a minimal valid settings.yaml to a temp path."""
    settings = {
        "defaults": {"rounds": 2, "output_dir": "./output", "synthesizer": "claude"},
        "models": {
            "claude": {
                "sdk": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "api_key_env": "TEST_CLAUDE_KEY",
                "timeout_sec": 60,
                "max_tokens": 4096,
            }
        },
        "prompts": {
            "initial": "Answer: {question}",
            "critique": "Round {round}. Q: {question}\n{previous_responses}",
            "synthesis": "Q: {question}\n{full_transcript}\n{rounds}",
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
    assert config.defaults.synthesizer == "claude"
    assert isinstance(config.defaults.output_dir, Path)


def test_load_config_models(minimal_settings):
    config = load_config(minimal_settings)
    assert "claude" in config.models
    assert isinstance(config.models["claude"], ModelConfig)
    assert config.models["claude"].model == "claude-sonnet-4-20250514"


def test_load_config_prompts(minimal_settings):
    config = load_config(minimal_settings)
    assert isinstance(config.prompts, PromptsConfig)
    assert "{question}" in config.prompts.initial


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
