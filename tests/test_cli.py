"""Tests for CLI panel/synthesizer selection logic in src/cli.py."""

import pytest

from src.cli import _determine_panel, _exclude_synthesizer_from_panel, _pick_non_participant_synthesizer
from tests.conftest import MockProvider


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
    panel, mode = _determine_panel(sample_app_config, models_arg="claude,openai", full_flag=False)
    assert panel == ["claude", "openai"]
    assert mode == "custom"


def test_determine_panel_models_arg_overrides_full(sample_app_config):
    """--models should override --full."""
    panel, mode = _determine_panel(sample_app_config, models_arg="claude,grok", full_flag=True)
    assert panel == ["claude", "grok"]
    assert mode == "custom"


def test_cli_full_flag_uses_full_panel(sample_app_config):
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
        mock_all_providers, panel_names, preferred="claude"  # claude is in panel
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
        all_providers, panel_names, preferred="openai"  # not available
    )
    assert is_participant is True
    assert synth.name() in {"claude", "gemini"}


# --- _exclude_synthesizer_from_panel tests ---

def test_exclude_synthesizer_from_full_panel():
    """Full panel including synthesizer: synthesizer removed, leaving 4 debaters."""
    all_providers = {n: MockProvider(n) for n in ["claude", "gemini", "deepseek", "openai", "grok"]}
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
